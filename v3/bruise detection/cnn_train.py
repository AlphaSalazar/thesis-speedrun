import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import cv2
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset

# Define dataset paths
DATASET_PATHS = {
    "train": "dataset/train",
    "val": "dataset/val",
    "test": "dataset/test"
}

# Custom dataset class with preprocessing
class MangoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.classes = ["bruised", "not_bruised"]

        for label, category in enumerate(self.classes):
            category_path = os.path.join(root_dir, category)
            if not os.path.exists(category_path):
                continue
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                self.data.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def preprocess_image(self, img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        equalized = cv2.equalizeHist(gray)  # Histogram equalization
        return equalized

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = self.preprocess_image(img_path)

        # Convert to tensor (1-channel)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Keep single channel

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load dataset with updated paths
train_dataset = MangoDataset(root_dir=DATASET_PATHS["train"], transform=transform)
val_dataset = MangoDataset(root_dir=DATASET_PATHS["val"], transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Load EfficientNet model and modify input layer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(weights="IMAGENET1K_V1")

# Modify first convolutional layer to accept grayscale (1-channel) input
old_conv = model.features[0][0]  # Get the first conv layer
new_conv = nn.Conv2d(in_channels=1,  # Change to single channel
                      out_channels=old_conv.out_channels, 
                      kernel_size=old_conv.kernel_size,
                      stride=old_conv.stride, 
                      padding=old_conv.padding, 
                      bias=old_conv.bias is not None)

# Copy weights (averaging RGB channels to grayscale)
new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)

# Replace first conv layer in EfficientNet
model.features[0][0] = new_conv

# Modify classifier for binary classification
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

# Move model to GPU
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with validation
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    # Validation step
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.2f}%")

# Save model
torch.save(model.state_dict(), 'mango_cnn.pth')
print("Model training complete and saved.")
