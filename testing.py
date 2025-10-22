import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from PIL import Image

# --- 1. Setup Dataloaders and Class Names ---
# (We need this to get the class names and the validation loader)

valid_dir = "G:/v2/Nail Disease DataSet/Test"
train_dir = "G:/v2/Nail Disease DataSet/Train" # Need this to get class mapping

# Use the *exact* same validation transforms as in training
valid_transforms = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load train data just to get the class names and number of classes
# This ensures the class indices match the model's output
train_data = ImageFolder(root=train_dir)
class_names = train_data.classes
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}")

# Create the validation loader
valid_data = ImageFolder(root=valid_dir, transform=valid_transforms)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False) # No shuffle for evaluation

# --- 2. Load Your Saved Model ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_save_path = "trained models/17classes_resnet_model_new.pth"

# Re-create the model architecture
model = timm.create_model('resnet18d', pretrained=False, num_classes=num_classes)

# Load the saved weights
# map_location ensures it works even if you trained on GPU and now use CPU
model.load_state_dict(torch.load(model_save_path, map_location=device))

model.to(device)
model.eval()  # *** CRITICAL: Set model to evaluation mode ***
              # This turns off dropout, uses batch norm running stats, etc.

print(f"Model loaded from {model_save_path} and set to eval mode on {device}.")


# --- 3. Get All Predictions ---

all_preds = []
all_labels = []

# Turn off gradients for evaluation
with torch.no_grad():
    for images, labels in valid_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Get the class with the highest score
        _, predicted = torch.max(outputs, 1)
        
        # Store predictions and labels
        # Move to CPU and convert to numpy for sklearn
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("Evaluation complete. Calculating metrics...")

# --- 4. Calculate and Print Metrics ---

# Overall Accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")

# Classification Report (Precision, Recall, F1-Score)
print("\nClassification Report:")
# target_names makes the report readable with your class names
report = classification_report(all_labels, all_preds, target_names=class_names)
print(report)

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(all_labels, all_preds)
print(cm)


# --- 5. Plot the Confusion Matrix (Optional but helpful) ---

# Since you have 17 classes, we need a larger figure
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# Found 17 classes: ['Darier_s disease', 'Muehrck-e_s lines', 'aloperia areata', 'beau_s lines', 'bluish nail', 'clubbing', 'eczema', 'half and half nailes (Lindsay_s nails)', 'koilonychia', 'leukonychia', 'onycholycis', 'pale nail', 'red lunula', 'splinter hemmorrage', 'terry_s nail', 'white nail', 'yellow nails']
# Model loaded from trained models/17classes_resnet_model_new.pth and set to eval mode on cpu.
# Evaluation complete. Calculating metrics...

# Overall Accuracy: 100.00%

# Classification Report:
#                                         precision    recall  f1-score   support

#                       Darier_s disease       1.00      1.00      1.00        17
#                      Muehrck-e_s lines       1.00      1.00      1.00         9
#                        aloperia areata       1.00      1.00      1.00        15
#                           beau_s lines       1.00      1.00      1.00         8
#                            bluish nail       1.00      1.00      1.00        13
#                               clubbing       1.00      1.00      1.00        12
#                                 eczema       1.00      1.00      1.00        12
# half and half nailes (Lindsay_s nails)       1.00      1.00      1.00        15
#                            koilonychia       1.00      1.00      1.00         8
#                            leukonychia       1.00      1.00      1.00         6
#                            onycholycis       1.00      1.00      1.00        12
#                              pale nail       1.00      1.00      1.00         8
#                             red lunula       1.00      1.00      1.00        15
#                    splinter hemmorrage       1.00      1.00      1.00        10
#                           terry_s nail       1.00      1.00      1.00         9
#                             white nail       1.00      1.00      1.00         6
#                           yellow nails       1.00      1.00      1.00         8

#                               accuracy                           1.00       183
#                              macro avg       1.00      1.00      1.00       183
#                           weighted avg       1.00      1.00      1.00       183


# Confusion Matrix:
# [[17  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
#  [ 0  9  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
#  [ 0  0 15  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
#  [ 0  0  0  8  0  0  0  0  0  0  0  0  0  0  0  0  0]
#  [ 0  0  0  0 13  0  0  0  0  0  0  0  0  0  0  0  0]
#  [ 0  0  0  0  0 12  0  0  0  0  0  0  0  0  0  0  0]
#  [ 0  0  0  0  0  0 12  0  0  0  0  0  0  0  0  0  0]
#  [ 0  0  0  0  0  0  0 15  0  0  0  0  0  0  0  0  0]
#  [ 0  0  0  0  0  0  0  0  8  0  0  0  0  0  0  0  0]
#  [ 0  0  0  0  0  0  0  0  0  6  0  0  0  0  0  0  0]
#  [ 0  0  0  0  0  0  0  0  0  0 12  0  0  0  0  0  0]
#  [ 0  0  0  0  0  0  0  0  0  0  0  8  0  0  0  0  0]
#  [ 0  0  0  0  0  0  0  0  0  0  0  0 15  0  0  0  0]
#  [ 0  0  0  0  0  0  0  0  0  0  0  0  0 10  0  0  0]
#  [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  9  0  0]
#  [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  6  0]
#  [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  8]]