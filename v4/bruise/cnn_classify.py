import torch
import cv2
import numpy as np
from efficientnet_pytorch import EfficientNet
import time

def preprocess_webcam_image(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.equalizeHist(frame)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    _, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    frame = frame / 255.0  # Normalize
    frame = np.expand_dims(frame, axis=0)  # Add channel dimension
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return torch.tensor(frame, dtype=torch.float32)

# Load model
model = EfficientNet.from_name('efficientnet-b0', in_channels=1, num_classes=2)
model.load_state_dict(torch.load("mango_cnn.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

def classify_mango():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    scores = []
    for i in range(2):  # Evaluate twice (top and bottom of mango)
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            continue
        
        input_tensor = preprocess_webcam_image(frame).to(device)
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
        scores.append(prediction + 1)  # Convert 0,1 to 1,2
        
        time.sleep(5)  # Wait before capturing second image
    
    cap.release()
    cv2.destroyAllWindows()
    final_score = sum(scores) / len(scores)
    print(f"Mango Grade: {final_score}")

if __name__ == "__main__":
    classify_mango()
