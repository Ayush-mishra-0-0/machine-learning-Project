import torch
import yaml
from models.yolo import Model
from utils.general import check_img_size, check_requirements, colorstr, increment_path
from utils.torch_utils import select_device
from tqdm import tqdm
from pathlib import Path
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from custom_dataset import PlantDiseaseDataset

# # Load configuration
# with open(r'data\plant_data.yaml') as f:
#     data_dict = yaml.safe_load(f)

# Hyperparameters
LEARNING_RATE = 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005
EPOCHS = 300
BATCH_SIZE = 16
IMG_SIZE = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# def train():
#     device = torch.device(DEVICE)
#     print(f"Using device: {device}")
    
#     # Load model
#     model = Model('models/yolov5s_psan.yaml', ch=3, nc=data_dict['nc']).to(device)
    
#     # Optimizer
#     optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    
#     # Scheduler
#     lf = lambda x: (1 - x / EPOCHS) * (1.0 - 0.01) + 0.01  # linear
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
#     # Custom Dataset and DataLoader
#     transform = transforms.Compose([
#         transforms.Resize((IMG_SIZE, IMG_SIZE)),
#         transforms.ToTensor(),
#     ])

#     dataset = PlantDiseaseDataset(
#         csv_file=r'C:\Users\Ayush\OneDrive\Documents\ml\krishi.ai\processed_data\FGVC8\train\train.csv',
#         img_dir=r'C:\Users\Ayush\OneDrive\Documents\ml\krishi.ai\processed_data\FGVC8',
#         transform=transform,
#         img_size=IMG_SIZE
#     )

#     train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)

#     # Start training
#     for epoch in range(EPOCHS):
#         model.train()
#         pbar = tqdm(enumerate(train_loader), total=len(train_loader))
#         for i, (imgs, labels) in pbar:
#             print(f"Batch shape: {imgs.shape}, Labels shape: {labels.shape}")
#             imgs = imgs.to(device)
#             labels = labels.to(device)

#             # Forward
#             outputs = model(imgs)

#             # Compute loss (you'll need to implement this based on YOLOv5's loss function)
#             loss = compute_loss(outputs, labels)

#             # Backward
#             loss.backward()

#             # Optimize
#             optimizer.step()
#             optimizer.zero_grad()

#             # Update progress bar
#             pbar.set_description(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss.item():.4f}")

#         # Update scheduler
#         scheduler.step()

#         # Save checkpoint every 10 epochs
#         if (epoch + 1) % 10 == 0:
#             checkpoint_path = f'checkpoints/yolov5_psan_epoch_{epoch+1}.pt'
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': loss,
#             }, checkpoint_path)
#             print(f"Checkpoint saved to {checkpoint_path}")

# def custom_collate_fn(batch):
#     images = torch.cat([item[0] for item in batch], dim=0)
#     labels = torch.cat([item[1] for item in batch], dim=0)
#     return images, labels

# def compute_loss(outputs, labels):
#     # Implement YOLOv5 loss function here
#     # This is a placeholder and needs to be replaced with the actual YOLOv5 loss computation
#     return torch.nn.functional.mse_loss(outputs, labels)

# if __name__ == '__main__':
#     check_requirements(exclude=('tensorboard', 'thop'))
#     train()


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# Import your PSAN model and other necessary modules
from psan import PSAN  # You'll need to implement this
from  custom_dataset import PlantDiseaseDataset

def train(opt):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data augmentation and normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = PlantDiseaseDataset(csv_file='C:/Users/Ayush/OneDrive/Documents/ml/krishi.ai/processed_data/FGVC8/train/train.csv', 
                                        img_dir='C:/Users/Ayush/OneDrive/Documents/ml/krishi.ai/processed_data/FGVC8', 
                                        transform=transform)
    val_dataset = PlantDiseaseDataset(csv_file='C:/Users/Ayush/OneDrive/Documents/ml/krishi.ai/processed_data/FGVC8/val/val.csv', 
                                      img_dir='C:/Users/Ayush/OneDrive/Documents/ml/krishi.ai/processed_data/FGVC8', 
                                      transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

    # Initialize the PSAN model
    model = PSAN(num_classes=12).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # Training loop
    for epoch in range(opt.epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{opt.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # Validation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Validation Accuracy: {100 * correct / total}%')

    # Save the model
    torch.save(model.state_dict(), 'psan_model.pth')

if __name__ == '__main__':
    class Opt:
        batch_size = 32
        workers = 4
        lr = 0.001
        epochs = 50

    opt = Opt()
    train(opt)