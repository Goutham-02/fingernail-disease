import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import os

# --- 1. Configuration & Setup ---

VALID_DIR = "G:/v2/Nail Disease DataSet/Test"
TRAIN_DIR = "G:/v2/Nail Disease DataSet/Train" # Need this to get class mapping
MODEL_PATH = "trained models/17classes_resnet_model_new.pth"
NUM_IMAGES = 8 # Number of images to display (e.g., 8 for a 2x4 grid)

# Check paths
assert os.path.exists(VALID_DIR), f"Validation directory not found: {VALID_DIR}"
assert os.path.exists(TRAIN_DIR), f"Training directory not found: {TRAIN_DIR}"
assert os.path.exists(MODEL_PATH), f"Model file not found: {MODEL_PATH}"

# --- 2. Load Data and Class Names ---

# Use the *exact* same validation transforms
valid_transforms = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load train data *only* to get the class_names mapping
train_data = ImageFolder(root=TRAIN_DIR)
class_names = train_data.classes
num_classes = len(class_names)
print(f"Found {num_classes} classes.")

# Load validation data and create a loader
valid_data = ImageFolder(root=VALID_DIR, transform=valid_transforms)
# Set shuffle=True to get random images easily
valid_loader = DataLoader(valid_data, batch_size=NUM_IMAGES, shuffle=True)

# --- 3. Load Model ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-create the model architecture
model = timm.create_model('resnet18d', pretrained=False, num_classes=num_classes)

# Load the saved weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.to(device)
model.eval() # Set to evaluation mode
print(f"Model loaded and set to eval mode on {device}.")

# --- 4. Grad-CAM Implementation ---

class GradCAM:
    """
    Implements Grad-CAM for a given model.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activations)
        # Use register_full_backward_hook for newer PyTorch versions
        if hasattr(self.target_layer, 'register_full_backward_hook'):
            self.target_layer.register_full_backward_hook(self.save_gradients)
        else:
            self.target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        # grad_output is a tuple, we want the first element
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, target_class=None):
        # --- Forward pass ---
        # Clear gradients just in case
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if target_class is None:
            # Use the predicted class if none is specified
            target_class = output.argmax(dim=1).item()
        
        # --- Backward pass ---
        # Get the score for the target class
        score = output[:, target_class]
        score.backward(retain_graph=True) # Backpropagate
        
        # --- Generate Heatmap ---
        if self.gradients is None or self.activations is None:
            print("Error: Gradients or activations not captured.")
            return None, None

        # Get the gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        # Global Average Pooling of gradients
        # (N, C, H, W) -> (N, C)
        weights = torch.mean(gradients, dim=[2, 3])
        
        # (N, C) -> (N, C, 1, 1) to broadcast
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        
        # Weighted sum of activations
        # (N, C, H, W) * (N, C, 1, 1) -> (N, C, H, W)
        weighted_activations = weights * activations
        
        # Sum across the channel dimension
        # (N, C, H, W) -> (N, 1, H, W)
        heatmap = torch.sum(weighted_activations, dim=1, keepdim=True)
        
        # Apply ReLU
        heatmap = F.relu(heatmap)
        
        # Normalize the heatmap
        heatmap_min, heatmap_max = heatmap.min(), heatmap.max()
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)
        
        return heatmap.squeeze(0), target_class

# --- 5. Helper Functions for Plotting ---

def denormalize_image(tensor):
    """Reverses the normalization for plotting."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device)
    # Reshape for broadcasting (C, 1, 1)
    mean = mean.view(3, 1, 1)
    std = std.view(3, 1, 1)
    
    # x = (x_norm * std) + mean
    tensor = tensor * std + mean
    # Clip to valid [0, 1] range
    tensor = torch.clamp(tensor, 0, 1)
    return tensor

def apply_heatmap(img_tensor, heatmap):
    """Applies the CAM heatmap to the original image."""
    # 1. Convert tensor image to numpy (H, W, C) for OpenCV
    img = denormalize_image(img_tensor).cpu().permute(1, 2, 0).numpy()
    img = np.uint8(img * 255)
    
    # 2. Resize heatmap and apply colormap
    heatmap_np = heatmap.cpu().detach().numpy()
    # (1, H, W) -> (H, W)
    heatmap_np = np.squeeze(heatmap_np) 
    
    # Resize heatmap to match image dimensions (192, 192)
    heatmap_resized = cv2.resize(heatmap_np, (img.shape[1], img.shape[0]))
    
    # Apply colormap
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
    
    # 3. Superimpose heatmap on original image
    superimposed_img = (heatmap_color * 0.4) + (img * 0.6)
    superimposed_img = np.uint8(np.clip(superimposed_img, 0, 255))
    
    return superimposed_img, img

# --- 6. Run Visualization ---

print("Generating Grad-CAM visualizations...")

# Find the target layer
# For resnet18d, 'model.layer4[-1].conv2' is a good choice (last conv layer)
target_layer = model.layer4[-1].conv2
grad_cam = GradCAM(model, target_layer)

# Get one batch of random images
images, labels = next(iter(valid_loader))
images = images.to(device)

# Create the plot
fig, axs = plt.subplots(2, 4, figsize=(20, 11)) # 2 rows, 4 columns
axs = axs.flatten() # Flatten for easy iteration

for i in range(NUM_IMAGES):
    if i >= len(images):
        # Hide any unused subplots
        axs[i].axis('off')
        continue
        
    img_tensor = images[i].unsqueeze(0) # (1, C, H, W)
    true_label = class_names[labels[i].item()]

    # Get Grad-CAM heatmap and predicted class index
    heatmap, pred_idx = grad_cam(img_tensor)
    pred_label = class_names[pred_idx]
    
    # Apply heatmap to image
    overlay, original_img = apply_heatmap(images[i], heatmap)
    
    # Set title color
    color = "green" if true_label == pred_label else "red"
    
    # Plot the overlay
    axs[i].imshow(overlay)
    axs[i].set_title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=12)
    axs[i].axis('off')

plt.suptitle(f"Grad-CAM Visualizations for {model.default_cfg['architecture']}", fontsize=16)
plt.tight_layout()
plt.show()

print("Visualization complete.")