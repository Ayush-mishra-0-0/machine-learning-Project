import torch
from torchvision import transforms
from custom_dataset import PlantDiseaseDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def test_dataset():
    # Define the transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    try:
        # Create the dataset
        dataset = PlantDiseaseDataset(
            csv_file=r'C:\Users\Ayush\OneDrive\Documents\ml\krishi.ai\processed_data\FGVC8\train\train.csv',
            img_dir=r'C:\Users\Ayush\OneDrive\Documents\ml\krishi.ai\processed_data\FGVC8',
            transform=transform
        )

        # Create a dataloader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        # Get a batch of samples
        images, labels = next(iter(dataloader))

        # Print some information
        print(f"Number of samples in dataset: {len(dataset)}")
        print(f"Number of classes: {len(dataset.label_map)}")
        print(f"Class mapping: {dataset.label_map}")
        print(f"Batch shape: {images.shape}")
        print(f"Labels in this batch: {labels}")

        # Display the images
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        for i in range(4):
            img = images[i].permute(1, 2, 0).numpy()
            ax = axs[i//2, i%2]
            ax.imshow(img)
            ax.set_title(f"Label: {list(dataset.label_map.keys())[labels[i].item()]}")
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    test_dataset()