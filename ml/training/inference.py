import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import os

# Add the project root to Python path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.models.model import get_model

def load_checkpoint(checkpoint_path, num_classes=10):
    """Load a trained model checkpoint"""

    # Create model instance
    model = get_model(num_classes=num_classes)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def predict_single_image(model, image_path, device='cpu'):
    """Predict a single image using the trained model"""

    # Set model to evaluation mode
    model.eval()

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Move to device
    img_tensor = img_tensor.to(device)

    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)

    return predicted_class.item(), probabilities[0].cpu().numpy()

def run_inference(checkpoint_path, image_path, num_classes=10):
    """Run inference on a single image"""

    print(f"Loading model from: {checkpoint_path}")
    model = load_checkpoint(checkpoint_path, num_classes)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    print(f"Running inference on: {image_path}")
    predicted_class, probabilities = predict_single_image(model, image_path, device)

    print(f"Predicted class: {predicted_class}")
    print(f"Probabilities: {probabilities}")

    return predicted_class, probabilities

if __name__ == "__main__":
    # Example usage - this would be called with specific paths
    print("Inference script loaded successfully")
    print("To run inference:")
    print("  python ml/training/inference.py --checkpoint-path path/to/checkpoint.pth --image-path path/to/image.jpg")
