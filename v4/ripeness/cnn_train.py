import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import cv2
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B0_Weights
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Define dataset paths
DATASET_PATHS = {
    "train": "C:/Users/dangi/OneDrive/Desktop/THESIS/v3/ripeness_detection/dataset/train",
    "val": "C:/Users/dangi/OneDrive/Desktop/THESIS/v3/ripeness_detection/dataset/val",
    "test": "C:/Users/dangi/OneDrive/Desktop/THESIS/v3/ripeness_detection/dataset/train"
}

# Custom dataset class
class MangoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.classes = ["Green", "Green_Yellow", "Yellow"]

        for label, category in enumerate(self.classes):
            category_path = os.path.join(root_dir, category)
            if not os.path.exists(category_path):
                print(f"Warning: {category_path} does not exist!")
                continue

            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):  
                    self.data.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def preprocess_image(self, img_path):
        img = cv2.imread(img_path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([10, 40, 40])
        upper_bound = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        img = cv2.bitwise_and(img, img, mask=mask)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return img

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = self.preprocess_image(img_path)
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

# Data transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load datasets
train_dataset = MangoDataset(root_dir=DATASET_PATHS["train"], transform=transform)
val_dataset = MangoDataset(root_dir=DATASET_PATHS["val"], transform=transform)
test_dataset = MangoDataset(root_dir=DATASET_PATHS["test"], transform=transform)

train_loader = DataLoader(train_dataset, batch_size=56, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=56, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=56, shuffle=False)

# Load EfficientNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        progress_bar.set_postfix(loss=running_loss / len(train_loader))
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

# Save model
torch.save(model.state_dict(), 'mango_cnn.pth')
print("Model saved successfully.")

# Evaluation on test set
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Classification report and confusion matrix
report = classification_report(y_true, y_pred, target_names=["Green", "Green_Yellow", "Yellow"], output_dict=True)
conf_matrix = confusion_matrix(y_true, y_pred)

# Convert report to DataFrame
report_df = pd.DataFrame(report).transpose()
print("\nClassification Report:")
print(report_df)

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Green", "Green_Yellow", "Yellow"], yticklabels=["Green", "Green_Yellow", "Yellow"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Print confusion matrix
print("\nConfusion Matrix:")
print(pd.DataFrame(conf_matrix, index=["Green", "Green_Yellow", "Yellow"], columns=["Green", "Green_Yellow", "Yellow"]))
