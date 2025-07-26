import os
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.utils.data import random_split, Subset
from torchvision.transforms import InterpolationMode

# 定义数据增强的转换
data_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),  # 显式指定双线性插值
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

root_dir = "GelFabric"

# 获取所有类别文件夹
categories = os.listdir(root_dir)

# 统计每个类别的图片数量
category_counts = {}
for category in categories:
    category_dir = os.path.join(root_dir, category)
    if os.path.isdir(category_dir):
        image_count = sum(len(files) for _, _, files in os.walk(category_dir))
        category_counts[category] = image_count

for category, count in category_counts.items():
    print(f"Category: {category}, Image Count: {count}")

# 加载数据集
original_dataset = []
for category_dir in os.listdir(root_dir):
    category_path = os.path.join(root_dir, category_dir)
    if not os.path.isdir(category_path):
        continue
    touch_results_dir = os.path.join(category_path, "tactile")
    vision_results_dir = os.path.join(category_path, "visual")
    if not os.path.isdir(touch_results_dir) or not os.path.isdir(vision_results_dir):
        continue
    touch_images = os.listdir(touch_results_dir)
    vision_images = os.listdir(vision_results_dir)
    for imag_id in range(len(touch_images)):
         touch_image_path = os.path.join(touch_results_dir, '({}).jpg'.format(imag_id+1))
            # 根据触摸图像名获取对应的视觉图像名
         vision_image_path = os.path.join(vision_results_dir, '({}).JPG'.format(imag_id+1))
         if os.path.isfile(vision_image_path):
            touch_image = Image.open(touch_image_path).convert('RGB')
            vision_image = Image.open(vision_image_path).convert('RGB')
            original_dataset.append((touch_image, vision_image, category_dir))


# 统计每个类别的图片数量
class_counts = {}
for _, _, label in original_dataset:
    class_counts[label] = class_counts.get(label, 0) + 1

max_class_count = max(class_counts.values())
target_class_counts = {label: max_class_count * 1 - count for label, count in class_counts.items()}

label_dict = {label: idx for idx, label in enumerate(os.listdir(root_dir))}

augmented_dataset = []
for touch_image, vision_image, label in original_dataset:
    augmented_dataset.append((data_transform(touch_image), data_transform(vision_image), label_dict[label]))
    if target_class_counts[label] > 0:
        for _ in range(target_class_counts[label]):
            transformed_touch_image = data_transform(touch_image)
            transformed_vision_image = data_transform(vision_image)
            augmented_dataset.append((transformed_touch_image, transformed_vision_image, label_dict[label]))
            target_class_counts[label] -= 1

def save_data(output_folder, dataset):
    # 创建目标文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, (touch_image, vision_image, label) in enumerate(dataset):
        class_name = list(label_dict.keys())[list(label_dict.values()).index(label)]
        class_folder = os.path.join(output_folder, class_name)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
            os.makedirs(os.path.join(class_folder, 'touch'))
            os.makedirs(os.path.join(class_folder, 'vision'))

        touch_image_path = os.path.join(class_folder, f"touch/{i}.npy")
        vision_image_path = os.path.join(class_folder, f"vision/{i}.npy")

        np.save(touch_image_path, touch_image.numpy())
        np.save(vision_image_path, vision_image.numpy())

# 将数据按类别分组
def split_dataset_by_class(dataset):
    class_to_indices = {}
    for idx, (touch_image, vision_image, label) in enumerate(dataset):
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(idx)
    return class_to_indices

# 将每个类别的数据分成训练、验证、测试集
def split_class_indices(class_indices, train_ratio, val_ratio):
    train_indices = []
    val_indices = []
    test_indices = []

    for label, indices in class_indices.items():
        class_size = len(indices)
        train_size = int(train_ratio * class_size)
        val_size = int(val_ratio * class_size)
        test_size = class_size - train_size - val_size

        train_indices.extend(indices[:train_size])
        val_indices.extend(indices[train_size:train_size + val_size])
        test_indices.extend(indices[train_size + val_size:])

    return train_indices, val_indices, test_indices

# 定义数据集划分比例
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 1 - train_ratio - val_ratio

# 按类别分组数据集
class_to_indices = split_dataset_by_class(augmented_dataset)

# 按类别分别划分数据集
train_indices, val_indices, test_indices = split_class_indices(class_to_indices, train_ratio, val_ratio)

# 使用Subset创建新的数据集
train_dataset = Subset(augmented_dataset, train_indices)
val_dataset = Subset(augmented_dataset, val_indices)
test_dataset = Subset(augmented_dataset, test_indices)

output_folder = "./data/"
# 保存数据
save_data(os.path.join(output_folder, "train"), train_dataset)
save_data(os.path.join(output_folder, "val"), val_dataset)
save_data(os.path.join(output_folder, "test"), test_dataset)

print("Data saved successfully!")
