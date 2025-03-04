import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import cv2
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B0_Weights

# Define dataset paths
DATASET_PATHS = {
    "train": "ripeness detection/dataset/train",
    "val": "ripeness detection/dataset/val",
    "test": "ripeness detection/dataset/test"
}

# Custom dataset class with preprocessing
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

            image_count = 0
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):  
                    self.data.append(img_path)
                    self.labels.append(label)
                    image_count += 1
            
            print(f"Loaded {image_count} images from {category_path}")

    def __len__(self):
        return len(self.data)

    def preprocess_image(self, img_path):
        img = cv2.imread(img_path)

        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Background removal using color thresholding
        lower_bound = np.array([10, 40, 40])   # Adjust based on mango color range
        upper_bound = np.array([40, 255, 255]) # Detect mango and remove background
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Apply mask to remove background
        img = cv2.bitwise_and(img, img, mask=mask)

        # Histogram equalization on V channel
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

        # Convert back to RGB
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
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize RGB
])

# Load dataset
train_dataset = MangoDataset(root_dir=DATASET_PATHS["train"], transform=transform)
val_dataset = MangoDataset(root_dir=DATASET_PATHS["val"], transform=transform)

train_loader = DataLoader(train_dataset, batch_size=56, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=56, shuffle=False)

# Load EfficientNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

# Modify classifier for 3-class classification
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 25
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
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_accuracy:.2f}%")

# Save model
torch.save(model.state_dict(), 'mango_cnn.pth')
print("Model training complete and saved.")
