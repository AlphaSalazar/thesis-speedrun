import cv2
import numpy as np  # <-- Add this line
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
model.load_state_dict(torch.load("mango_cnn.pth"))
model.to(device)
model.eval()

# Preprocessing function
def preprocess_image(frame):
    """Apply background removal, histogram equalization, and normalization."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Background removal using color thresholding
    lower_bound = np.array([10, 40, 40])   # Adjust based on mango color range
    upper_bound = np.array([40, 255, 255]) 
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Apply mask to remove background
    frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Histogram equalization on V channel
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

    # Convert back to RGB
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize RGB
    ])
    
    img = transform(frame).unsqueeze(0).to(device)  # Add batch dimension
    return img

# Capture and classify images from webcam
def classify_mango():
    cap = cv2.VideoCapture(0)
    scores = []

    for i in range(2):  
        input("Position the mango and press Enter to capture...")
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            continue
        
        img = preprocess_image(frame)

        output = model(img)
        _, predicted = torch.max(output, 1)
        scores.append(predicted.item() + 1)  

        if i == 0:
            print("Evaluating bottom side... Please rotate mango.")
        cv2.waitKey(5000) 

    cap.release()
    cv2.destroyAllWindows()

    avg_score = sum(scores) / len(scores)
    print(f"Final Mango Ripeness Score: {avg_score:.1f}")

if __name__ == "__main__":
    classify_mango()
