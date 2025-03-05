import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import cv2
import numpy as np
from efficientnet_pytorch import EfficientNet
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Define dataset paths
DATASET_PATHS = {
    "train": "C:/Users/dangi/OneDrive/Desktop/THESIS/v4/bruise/dataset/train",
    "val": "C:/Users/dangi/OneDrive/Desktop/THESIS/v4/bruise/dataset/val",
    "test": "C:/Users/dangi/OneDrive/Desktop/THESIS/v4/bruise/dataset/test"
}

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Resize all images to 224x224
    image = cv2.resize(image, (224, 224))

    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add channel dimension
    return torch.tensor(image, dtype=torch.float32)


# Custom dataset to apply preprocessing
class MangoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        for label, category in enumerate(["bruised", "unbruised"]):
            category_path = os.path.join(root_dir, category)
            for img in os.listdir(category_path):
                self.image_paths.append(os.path.join(category_path, img))
                self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = preprocess_image(self.image_paths[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

# Load datasets
def get_data_loader(dataset_path, batch_size=64):
    dataset = MangoDataset(dataset_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_loader = get_data_loader(DATASET_PATHS['train'])
val_loader = get_data_loader(DATASET_PATHS['val'])
test_loader = get_data_loader(DATASET_PATHS['test'])

# Define model
model = EfficientNet.from_name('efficientnet-b0', in_channels=1, num_classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, epochs=50):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
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
            progress_bar.set_postfix(loss=running_loss / (total // images.size(0)), accuracy=correct / total)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {running_loss/len(train_loader):.4f}, Accuracy: {correct/total:.4f}")
    torch.save(model.state_dict(), "mango_cnn.pth")
    print("Model saved successfully.")

def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["bruised", "unbruised"]))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["bruised", "unbruised"], yticklabels=["bruised", "unbruised"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

train_model(model, train_loader, val_loader)
evaluate_model(model, test_loader)
