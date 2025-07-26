import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GraphConv
from torch_scatter import scatter_mean

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
        fused = vis_feat + tac_feat    # (B,512,7,7)
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
