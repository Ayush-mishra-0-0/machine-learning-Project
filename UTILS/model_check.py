import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# Load the pre-trained ResNet-101 model
model = models.resnet101(pretrained=False)

# Modify the final fully connected layer
num_ftrs = model.fc.in_features
num_classes = 12  # Number of classes in your fine-tuned model
model.fc = torch.nn.Linear(num_ftrs, num_classes)

# Load the state dict
state_dict = torch.load(r'C:\Users\Ayush\OneDrive\Documents\ml\krishi.ai\final_fine_tuned_resnet101.pth')

# Load the modified state dict
model.load_state_dict(state_dict)
model.eval()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the class labels
class_labels = [
    'healthy',
    'scab frog_eye_leaf_spot complex',
    'scab',
    'complex',
    'rust',
    'frog_eye_leaf_spot',
    'powdery_mildew',
    'scab frog_eye_leaf_spot',
    'frog_eye_leaf_spot complex',
    'rust frog_eye_leaf_spot',
    'powdery_mildew complex',
    'rust complex'
]

def predict_image(image_path):
    # Open and preprocess the image
    img = Image.open(image_path)
    
    # Apply the preprocessing transform
    img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
    
    # Move tensor to the appropriate device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    img_tensor = img_tensor.to(device)
    
    # Make a prediction
    with torch.no_grad():
        output = model(img_tensor)
    
    # Get the predicted class
    _, predicted = torch.max(output, 1)
    
    # Map the predicted class index to the actual class label
    result = class_labels[predicted.item()]
    
    return result

# Test the model on a single image
image_path = r'C:\Users\Ayush\OneDrive\Documents\ml\krishi.ai\data\FGVC8\train_images\8a0f52e99d52cd72.jpg'  
prediction = predict_image(image_path)
print(f"The predicted disease for the image is: {prediction}")