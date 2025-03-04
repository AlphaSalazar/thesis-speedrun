import cv2
import matplotlib.pyplot as plt

# Load the example image
image = cv2.imread('c:/Users/dangi/OneDrive/Desktop/THESIS/v3/ripeness detection/mango.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for proper visualization

# Check if the image was loaded correctly
if image is None:
    print("Error: Could not load image.")
else:
    # Normalize the image
    image_normalized = image / 255.0

    # Plot the original and normalized images side-by-side
    plt.figure(figsize=(10, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')

    # Normalized Image
    plt.subplot(1, 2, 2)
    plt.title('Normalized Image')
    plt.imshow(image_normalized)
    plt.axis('off')

    plt.show()
