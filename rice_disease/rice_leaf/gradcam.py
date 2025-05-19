# hàm gradcam dùng để trực quan hóa hình ảnh 
# để hiểu cách mạng nơ-ron tích chập được điều khiển để đưa ra quyết định phân loại
import torch 
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from torchvision import models, transforms
import streamlit as st
from tensorflow.keras.models import load_model

def get_model():
    return load_model("rice_disease/model/rice_model.h5") 

def get_model():
    model = models.resnet50(pretrained=True)
    # Modify the model as needed for your specific use case
    return model

def generate_gradcam(image_path):
    # Load and preprocess image
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0)
    
    # Get model predictions
    model = get_model()
    model.eval()
    
    # Generate Grad-CAM
    features = None
    gradient = None
    
    def hook_features(module, input, output):
        nonlocal features
        features = output
        
    def hook_grads(module, grad_in, grad_out):
        nonlocal gradient
        gradient = grad_out[0]
    
    # Register hooks
    model.layer4.register_forward_hook(hook_features)
    model.layer4.register_backward_hook(hook_grads)
    
    output = model(input_tensor)
    score = output[0].max()
    score.backward()
    
    pooled_gradients = torch.mean(gradient, dim=[0, 2, 3])
    for i in range(features.shape[1]):
        features[:, i, :, :] *= pooled_gradients[i]
    
    heatmap = torch.mean(features, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap = heatmap.detach().numpy()
    
    # Normalize heatmap
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    # Convert to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay on original image
    original_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    
    return Image.fromarray(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))