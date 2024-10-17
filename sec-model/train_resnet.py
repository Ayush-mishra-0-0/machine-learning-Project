import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

# Load the pre-trained ResNet-101 model
model = torchvision.models.resnet101()

# Load the state dict
path = r"C:\Users\Ayush\OneDrive\Desktop\machine-learning_Project\pre-trained_models\ResNet_101-ImageNet-model-99.pth"
state_dict = torch.load(path)

# Remove the 'fc' keys from the state dict
state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}

# Load the modified state dict
model.load_state_dict(state_dict, strict=False)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
num_classes = 12  
model.fc = torch.nn.Linear(num_ftrs, num_classes)

class CustomDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file)
        self.label_mapping = {label: idx for idx, label in enumerate(self.data['labels'].unique())}

        self.image_filenames = self.data['image'].values
        self.labels = self.data['labels'].map(self.label_mapping).values  

        self.valid_indices = []
        for idx, filename in enumerate(self.image_filenames):
            img_path = os.path.join(self.root_dir, filename)
            if os.path.exists(img_path):
                self.valid_indices.append(idx)
            else:
                print(f"Warning: Image not found: {img_path}")

        print(f"Found {len(self.valid_indices)} valid images out of {len(self.image_filenames)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        valid_idx = self.valid_indices[idx]
        img_name = os.path.join(self.root_dir, self.image_filenames[valid_idx])

        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name}: {str(e)}")
            image = Image.new('RGB', (256, 256), color='gray')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[valid_idx], dtype=torch.long)  # Use numeric labels

        return image, label


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = CustomDataset(
    root_dir=r'C:\Users\Ayush\OneDrive\Desktop\machine-learning_Project\data\FGVC8\train_images', 
    csv_file=r'C:\Users\Ayush\OneDrive\Desktop\machine-learning_Project\data\FGVC8\train.csv', 
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print(dataset.label_mapping)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Print whether we are using GPU or CPU
print(f"Training on: {device}")

# Early stopping parameters
patience = 5
min_delta = 0.001
patience_counter = 0
best_loss = float('inf')

num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    
    running_loss = 0.0
    
    # Use tqdm to add a progress bar
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch+1}/{num_epochs}')
    
    for batch_idx, (inputs, labels) in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        # Update tqdm bar with the current loss
        progress_bar.set_postfix({'Loss': loss.item()})
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss {loss.item():.4f}")
    
    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1} completed, Average Loss: {epoch_loss:.4f}")
    
    # Early stopping check
    if epoch_loss < best_loss - min_delta:
        best_loss = epoch_loss
        patience_counter = 0
        # Save the best model
        torch.save(model.state_dict(), 'best_fine_tuned_resnet101.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# Save the final trained model
torch.save(model.state_dict(), 'final_fine_tuned_resnet101.pth')