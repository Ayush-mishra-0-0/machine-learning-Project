import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class PlantDiseaseDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.csv_file= csv_file
        self.img_dir = img_dir
        self.transform = transform
        self.label_map = {label: idx for idx, label in enumerate(self.annotations['labels'].unique())}
        print(f"Dataset initialized with {len(self.annotations)} samples")
        print(f"First few rows of CSV:\n{self.annotations.head()}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = self.annotations.iloc[index, 0]
        if os.path.basename(self.csv_file) == 'processed_train.csv': 
            img_path = os.path.join(self.img_dir, 'train', img_name)
        else:
            img_path = os.path.join(self.img_dir, img_name)
        print(f"Attempting to open image at: {img_path}")
        
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            parent_dir = os.path.dirname(os.path.dirname(img_path))
            print(f"Contents of {parent_dir}:")
            print(os.listdir(parent_dir))
            raise FileNotFoundError(f"No file found at {img_path}")
        
        image = Image.open(img_path).convert("RGB")
        y_label = self.label_map[self.annotations.iloc[index, 1]]
        
        if self.transform:
            image = self.transform(image)
        
        return image, y_label