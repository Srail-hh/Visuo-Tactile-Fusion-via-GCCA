import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import GridEdgeGenerator, FeatureToGraph, MultiModalGC, DualModalModel
from models import ResNet18FeatureExtractor, GroupAttention
from datasets import CustomDataset
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# 超参数
PATH = 'best_fusion_model.pth'
# 数据加载
transform = None  # You can define your own transformations here if needed
test_dataset = CustomDataset('data/test')
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DualModalModel(num_classes = 6)
model.to(device)
model.load_state_dict(torch.load(PATH))
model.eval()

model.eval()
correct = 0
total = 0

predictions = []
true_labels = []

with torch.no_grad():
    for touch_image, vision_image, label in test_loader:
        touch_image, vision_image, label = touch_image.to(device), vision_image.to(device), label.to(device)

        outputs = model(vision_image, touch_image, generate_fake=False)
        _, predicted = torch.max(outputs.data, 1)

        total += label.size(0)
        correct += (predicted == label).sum().item()

        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(label.cpu().numpy())

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# 计算混淆矩阵
conf_matrix = confusion_matrix(true_labels, predictions)

# 可视化混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(6), yticklabels=range(6))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('Confusion_Matrix.png')
