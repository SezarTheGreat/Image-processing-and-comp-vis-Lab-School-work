import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Import libraries ---
# (Already done above)

# --- Step 2: Load input image ---
# Replace 'chessboard.jpg' with your image path. 
# Geometric shapes or building blocks work best for this demo.
image_path = 'boochan.png'
img = cv2.imread(image_path)

# Check if image loaded
if img is None:
    print(f"Error: Could not load image from {image_path}")
else:
    # Make a copy for display purposes later
    img_display = img.copy()
    
    # --- Step 3: Convert to Grayscale ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Step 4: Convert to float32 ---
    # Harris detector requires float32 input to calculate gradients accurately
    gray = np.float32(gray)

    # --- Step 5: Apply Harris Corner Detection ---
    # cv2.cornerHarris(input_image, blockSize, ksize, k)
    # blockSize (2): Neighborhood size
    # ksize (3): Aperture parameter for Sobel derivative
    # k (0.04): Harris detector free parameter
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    # --- Step 6: Dilate the result ---
    # This is purely for visualization; it makes the corner points "thicker"
    # so they are easier to see on the final image.
    dst = cv2.dilate(dst, None)

    # --- Step 7: Mark detected corners ---
    # We define a threshold. If the response is > 1% of the max response, 
    # we consider it a strong corner.
    threshold = 0.01 * dst.max()
    
    # Mark these points in Red [0, 0, 255] (Remember OpenCV is BGR)
    img[dst > threshold] = [0, 0, 255]

    # --- Step 8: Display Results ---
    plt.figure(figsize=(10, 5))

    # Convert BGR to RGB for Matplotlib display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.imshow(img_rgb)
    plt.title('Harris Corner Detection')
    plt.axis('off')
    plt.show()

    # --- Step 9: Save final output ---
    cv2.imwrite('harris_corners_output.jpg', img)
    print("Success: Output saved as 'harris_corners_output.jpg'")

# --- Step 10: End Program ---