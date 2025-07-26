import os
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))

        self.data_paths = []
        self.targets = []

        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            touch_dir = os.path.join(class_dir, 'touch')
            vision_dir = os.path.join(class_dir, 'vision')

            touch_files = os.listdir(touch_dir)
            vision_files = os.listdir(vision_dir)

            # Assuming each touch file has a corresponding vision file
            assert len(touch_files) == len(vision_files)

            for i in range(len(touch_files)):
                touch_path = os.path.join(touch_dir, touch_files[i])
                vision_path = os.path.join(vision_dir, touch_files[i])

                self.data_paths.append((touch_path, vision_path))
                self.targets.append(class_idx)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        touch_path, vision_path = self.data_paths[idx]
        target = self.targets[idx]

        # Load touch and vision data
        touch_data = np.load(touch_path)
        vision_data = np.load(vision_path)

        # Apply transformations if provided
        if self.transform:
            touch_data = self.transform(touch_data)
            vision_data = self.transform(vision_data)

        return touch_data, vision_data, target

