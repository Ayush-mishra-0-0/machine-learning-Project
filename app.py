from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import torch
import torchvision.models as models
from torchvision import transforms
from flask import send_from_directory
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained ResNet-101 model
model = models.resnet101(pretrained=False)
num_ftrs = model.fc.in_features
num_classes = 12
model.fc = torch.nn.Linear(num_ftrs, num_classes)

# Load the state dict
state_dict = torch.load(r'pre-trained_models/final_fine_tuned_resnet101.pth')
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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    img = Image.open(image_path)
    img_tensor = preprocess(img).unsqueeze(0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
    
    _, predicted = torch.max(output, 1)
    result = class_labels[predicted.item()]
    
    return result

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/model')
def model_page():
    return render_template('model.html')

@app.route('/dataset')
def dataset():
    return  render_template('dataset.html')

@app.route('/find')
def find():
    return render_template('find.html')


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'doc', 'docx','svg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the uploaded image
        result = predict_image(file_path)
        
        return render_template('result.html', result=result, image_filename=filename)
    return 'Invalid file type'

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5001)