import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GraphConv
from torch_scatter import scatter_mean

# -----------------------------------------------------------------------------
#  网格相关工具
# -----------------------------------------------------------------------------

def _generate_grid_edge_index(height: int, width: int) -> torch.Tensor:
    """生成 H×W 网格的邻接边。

    返回
    ----
    edge_index: LongTensor [2, num_edges]
        COO 形式的边索引。
    """
    edge_list = []
    for row in range(height):
        for col in range(width):
            node_id = row * width + col
            # 右邻接
            if col < width - 1:
                edge_list.append([node_id, node_id + 1])
            # 下邻接
            if row < height - 1:
                edge_list.append([node_id, node_id + width])
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()



class GridEdgeGenerator(nn.Module):


    def __init__(self, height: int = 7, width: int = 7):
        super().__init__()
        self.height = height
        self.width = width
        self.edge_index = _generate_grid_edge_index(height, width)



class FeatureToGraph(nn.Module):
    """把视觉 + 触觉特征映射成 PyG Batch 图。"""

    def __init__(self, visual_dim: int = 512, tactile_dim: int = 512, h: int = 7, w: int = 7):
        super().__init__()
        self.h, self.w = h, w
        self.projection = nn.Linear(visual_dim + tactile_dim, 2)
        # 预生成网格连接，注册为 buffer 以便自动迁移设备
        self.register_buffer("edge_index", _generate_grid_edge_index(h, w))

    def forward(self, visual_feat: torch.Tensor, tactile_feat: torch.Tensor) -> Batch:
        B, _, H, W = visual_feat.shape  # H=W=7
        device = visual_feat.device
        # 拼接通道，重排到 (B, 49, 1024)
        node_feat = torch.cat([visual_feat, tactile_feat], dim=1).permute(0, 2, 3, 1).reshape(B, H*W, -1)
        # 投影到 2D 坐标用于计算边权重
        coords = self.projection(node_feat)  # (B, 49, 2)

        base_edge = self.edge_index.to(device)  # [2, E]
        data_list = []
        for i in range(B):
            src, dst = base_edge
            dist = torch.norm(coords[i, src] - coords[i, dst], dim=1)  # [E]
            edge_attr = torch.sigmoid(1.0 / (dist + 1e-6)).unsqueeze(1)  # [E, 1]
            data_list.append(Data(x=node_feat[i], edge_index=base_edge, edge_attr=edge_attr))
        return Batch.from_data_list(data_list)


# -----------------------------------------------------------------------------
# 图卷积融合模块
# -----------------------------------------------------------------------------

class MultiModalGC(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512):
        super().__init__()
        self.graph_builder = FeatureToGraph(input_dim//2, input_dim//2)
        self.gc1 = GraphConv(input_dim, hidden_dim, aggr="mean")
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gc2 = GraphConv(hidden_dim, hidden_dim, aggr="mean")
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.alpha = 1
        # 解码器：将节点特征 + 边权重拼接后映射回 512 维
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, visual_feat: torch.Tensor, tactile_feat: torch.Tensor) -> torch.Tensor:
        # 1) 构建图
        graph = self.graph_builder(visual_feat, tactile_feat)
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        # 2) 边权重归一化
        alpha = torch.exp(edge_attr.squeeze()).view(-1)
        alpha = alpha / (alpha.sum() + 1e-8)
        # 3) 两层 GraphConv
        x = F.relu(self.bn1(self.gc1(x, edge_index, edge_weight=alpha)))
        x = F.relu(self.bn2(self.gc2(x, edge_index, edge_weight=alpha)))
        # 4) 将边属性按节点取均值
        edges = torch.cat([edge_index[0], edge_index[1]])
        edge_attr_rep = torch.cat([edge_attr, edge_attr], dim=0)
        node_w = scatter_mean(edge_attr_rep, edges, dim=0, dim_size=x.size(0))  # (N,1)
        # 5) 拼接并映射回 512 维
        x = self.decoder(torch.cat([x, node_w], dim=1))
        # 6) 恢复到 (B, 512, 7, 7)
        B = visual_feat.size(0)
        x = x.view(B, 49, 512).view(B, 7, 7, 512).permute(0, 3, 1, 2).contiguous()
        return x


# -----------------------------------------------------------------------------
# 分组注意力模块
# -----------------------------------------------------------------------------

class GroupAttention(nn.Module):
    def __init__(self, groups: int = 32, group_channels: int = 16):
        super().__init__()
        assert groups * group_channels == 512, "groups×group_channels 必须等于 512"
        self.groups = groups
        self.group_channels = group_channels
        self.conv = nn.Conv2d(group_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,512,7,7)
        B, C, H, W = x.shape
        assert C == self.groups * self.group_channels
        att_maps = []
        for g in range(self.groups):
            chunk = x[:, g*self.group_channels:(g+1)*self.group_channels]
            att_maps.append(torch.sigmoid(self.conv(chunk)))  # (B,1,7,7)
        return torch.cat(att_maps, dim=1)  # (B,groups,7,7)



class ResNet18FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-2])  # 输出(B,512,7,7)

    def forward(self, x):
        return self.features(x)


# -----------------------------------------------------------------------------
#  双模态分类网络
# -----------------------------------------------------------------------------

class DualModalModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # 两个模态的 ResNet18 backbone
        self.visual_backbone = ResNet18FeatureExtractor()
        self.tactile_backbone = ResNet18FeatureExtractor()
        # 图卷积融合
        self.fusion_gc = MultiModalGC(1024, 512)
        # 注意力
        self.visual_att = GroupAttention()
        self.tactile_att = GroupAttention()
        # 分类器 (输入 512 维全局池化向量)
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    # 内部：应用两组注意力并合并
    def _apply_attention(self, fused: torch.Tensor, vis_att: torch.Tensor, tac_att: torch.Tensor):
        B = fused.size(0)
        fused_g = fused.view(B, 32, 16, 7, 7)
        att = (fused_g * vis_att.unsqueeze(2) + fused_g * tac_att.unsqueeze(2)).view(B, 512, 7, 7)
        return att

    def forward(self, visual_img: torch.Tensor, tactile_img: torch.Tensor, *, generate_fake: bool = True):
        # 1) 提取特征
        vis_feat = self.visual_backbone(visual_img)   # (B,512,7,7)
        tac_feat = self.tactile_backbone(tactile_img) # (B,512,7,7)
        # 2) 特征融合
        fused = self.fusion_gc(vis_feat, tac_feat)    # (B,512,7,7)
        # 3) 注意力
        vis_att = self.visual_att(vis_feat)           # (B,32,7,7)
        tac_att = self.tactile_att(tac_feat)          # (B,32,7,7)
        att_fused = self._apply_attention(fused, vis_att, tac_att)  # (B,512,7,7)

        # helper: 全局均值池
        pool = lambda x: F.adaptive_avg_pool2d(x,1).view(x.size(0), -1)
        real_pool = pool(att_fused)
        y_real = self.classifier(real_pool)

        if not generate_fake:
            return y_real  # 推理阶段直接返回
        else:
        # 4) 生成随机假注意力
            fake_att = torch.rand_like(vis_att)
            att_fake = self._apply_attention(fused, fake_att, fake_att)
            fake_pool = pool(att_fake)
            y_fake = self.classifier(fake_pool)
        # 5) 对抗损失输出：真实减去假分类得分
            y_true = y_real - y_fake
            return y_true, y_real, y_fake
