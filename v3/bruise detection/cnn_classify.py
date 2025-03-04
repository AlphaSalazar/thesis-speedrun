import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights  # Correct import for weights

# Load pre-trained EfficientNet model with correct weight argument
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)  # Corrected method
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # Modify for binary classification

# Modify the first convolutional layer to accept 1-channel input
old_conv = model.features[0][0]  # Get the first convolutional layer
new_conv = nn.Conv2d(in_channels=1,  # Change to single channel
                      out_channels=old_conv.out_channels, 
                      kernel_size=old_conv.kernel_size,
                      stride=old_conv.stride, 
                      padding=old_conv.padding, 
                      bias=old_conv.bias is not None)

# Copy the weights from the original 3-channel first conv layer to the new 1-channel conv layer
new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)

# Replace the first conv layer in EfficientNet with the modified one
model.features[0][0] = new_conv

# Move the model to the appropriate device (GPU or CPU)
model = model.to(device)
model.eval()

# Preprocessing function for input frame
def preprocess_image(frame):
    """Convert the input frame to grayscale, apply histogram equalization, and normalize."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    equalized = cv2.equalizeHist(gray)  # Apply histogram equalization

    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    img = transform(equalized).unsqueeze(0).to(device)  # Add batch dimension -> Shape: (1, 1, 224, 224)
    return img

# Capture and classify images from webcam
def classify_mango():
    cap = cv2.VideoCapture(0)
    scores = []

    for i in range(2):  # Capture twice (top and bottom)
        input("Position the mango and press Enter to capture...")
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            continue
        
        # Preprocess the image
        img = preprocess_image(frame)

        # Pass through CNN model
        output = model(img)
        _, predicted = torch.max(output, 1)
        scores.append(predicted.item() + 1)  # Convert (0->1, 1->2)

        if i == 0:
            print("Evaluating bottom side... Please rotate mango.")
        cv2.waitKey(5000)  # 5-second delay

    cap.release()
    cv2.destroyAllWindows()

    # Compute average grade
    avg_score = sum(scores) / len(scores)
    print(f"Final Mango Score: {avg_score:.1f}")

if __name__ == "__main__":
    classify_mango()
