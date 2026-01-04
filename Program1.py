import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Step 1: Define inputs ---
# Replace 'sample_image.jpg' with the path to your actual image file
image_path = 'boochan.png'

# --- Step 2: Load input image ---
# cv2.imread loads the image in BGR format by default
img_bgr = cv2.imread(image_path)

# Check if the image was loaded successfully
if img_bgr is None:
    print(f"Error: Could not open or find the image at '{image_path}'.")
    print("Please check the file path and try again.")
else:
    print(f"Image loaded successfully. Dimensions: {img_bgr.shape}")

    # Setup matplotlib figure for displaying multiple steps
    plt.figure(figsize=(12, 8))

    # --- Step 3: Convert BGR to RGB ---
    # Essential because matplotlib expects RGB, but OpenCV provides BGR
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- Step 4: Display original RGB image ---
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original (RGB)")
    plt.axis('off') # Hide axis ticks

    # --- Step 5: Convert to Grayscale ---
    # Reduces image to a single channel (luminance only)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # --- Step 6: Display Grayscale image ---
    plt.subplot(1, 3, 2)
    # cmap='gray' is required to tell matplotlib to render it as black and white
    plt.imshow(img_gray, cmap='gray')
    plt.title("Grayscale")
    plt.axis('off')

    # --- Step 7: Resize the original image ---
    # Resizing to 300 x 300 pixels
    target_dim = (300, 300)
    img_resized = cv2.resize(img_bgr, target_dim)

    # --- Step 8: Display Resized image ---
    # Note: We must convert the resized BGR image to RGB for correct display
    img_resized_display = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    plt.subplot(1, 3, 3)
    plt.imshow(img_resized_display)
    plt.title(f"Resized {target_dim}")
    plt.axis('off')

    # Show all plots
    plt.show()

    # --- Step 9: Save processed images ---
    # cv2.imwrite expects BGR (for color) or single channel (gray)
    cv2.imwrite('output_grayscale.jpg', img_gray)
    cv2.imwrite('output_resized.jpg', img_resized)
    
    print("Processing complete. Images saved to disk.")