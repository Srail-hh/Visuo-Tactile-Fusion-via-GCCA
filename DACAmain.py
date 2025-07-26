import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from DAIAmodel import DualModalModel
from DAIAmodel import ResNet18FeatureExtractor, GroupAttention
from datasets import CustomDataset
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 使用GPU进行训练，如果没有GPU则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载器
train_dataset = CustomDataset('data/train')
val_dataset = CustomDataset('data/val')
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_dataset = CustomDataset('data/test')
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 模型路径和定义
PATH = 'best_fusion_model.pth'
model = DualModalModel(num_classes=6)  # 假设有n个类别
model.to(device)  # 将模型移动到GPU或CPU
torch.autograd.set_detect_anomaly(True)
# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)


# 训练设置  
num_epochs = 160
best_accuracy = 0.0  # 初始化最佳准确率

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss, correct, total = 0.0, 0, 0  # 初始化损失、正确预测和总样本数

    # 训练过程
    for inputs_touch, inputs_vision, labels in train_loader:
        inputs_touch, inputs_vision, labels = inputs_touch.to(device), inputs_vision.to(device), labels.to(device)

        optimizer.zero_grad()  # 清零梯度

        # 前向传播
        y_true, y_real, y_fake = model(inputs_touch, inputs_vision, generate_fake=True)
        
        # 计算损失
        ce_loss = criterion(y_true, labels)
        ly_loss = criterion(y_real, labels)
        total_loss = ce_loss + ly_loss

        # 反向传播和优化
        total_loss.backward()
        optimizer.step()
        
        # 统计损失和准确率
        running_loss += total_loss.item()
        _, predicted = torch.max(y_real, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # 输出本轮训练的损失和准确率
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {(correct/total)*100:.2f}%")
 

    # 在验证集上评估模型
    model.eval()  # 设置模型为评估模式
    correct, total = 0, 0
    predictions = []
    true_labels = []
    with torch.no_grad():  # 不计算梯度
        for touch_image, vision_image, label in val_loader:
            touch_image, vision_image, label = touch_image.to(device), vision_image.to(device), label.to(device)
            outputs = model(touch_image, vision_image, generate_fake=False)
            _, predicted = torch.max(outputs.data, 1)

            total += label.size(0)
            correct += (predicted == label).sum().item()
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(label.cpu().numpy())
    accuracy = 100 * correct / total  # 计算验证集准确率
    recall = recall_score(true_labels, predictions, average='macro')
    print(f"Validation Accuracy after epoch {epoch+1}: {accuracy:.2f}%")

    # 保存最佳模型
    if accuracy >= best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), PATH)
        print(f"Best model saved with accuracy: {best_accuracy:.2f}%")
        print(f"recall: {recall:.4f}")
        total_params = count_parameters(model)
        print(f"Total number of trainable parameters: {total_params}")
        # 计算混淆矩阵
        conf_matrix = confusion_matrix(true_labels, predictions)

        # 可视化混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(6), yticklabels=range(6))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.savefig(f'Confusion_Matrix{epoch+1}.png')
    # 每10个epoch保存一次checkpoint
    if (epoch + 1) % 10 == 0:
        test_correct, test_total = 0, 0
        test_predictions = []
        test_true_labels = []
        checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
        point_path = f'{epoch+1}.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / len(train_loader),
            'accuracy': accuracy
        }, checkpoint_path)
        torch.save(model.state_dict(), point_path)
        print(f"Checkpoint saved at epoch {epoch+1} to {checkpoint_path}")
        with torch.no_grad():  # 不计算梯度
            for touch_image, vision_image, label in test_loader:
                touch_image, vision_image, label = touch_image.to(device), vision_image.to(device), label.to(device)
                outputs = model(touch_image, vision_image, generate_fake=False)
                _, predicted = torch.max(outputs.data, 1)

                test_total += label.size(0)
                test_correct += (predicted == label).sum().item()
                test_predictions.extend(predicted.cpu().numpy())
                test_true_labels.extend(label.cpu().numpy())
        accuracy = 100 * test_correct / test_total  # 计算验证集准确率
        recall = recall_score(test_true_labels, test_predictions, average='macro')
        precision = precision_score(test_true_labels, test_predictions, average='macro')

        print(f"test Accuracy after epoch {epoch+1}: {accuracy:.2f}%")
        print(f"test recall: {recall:.4f}")
        print(f"test precision: {precision:.4f}")
 # 计算混淆矩阵
        conf_matrix = confusion_matrix(test_true_labels, test_predictions)

        # 可视化混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(6), yticklabels=range(6))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.savefig(f'test_Confusion_Matrix{epoch+1}.png')


print("completed!")
