import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- Step 1: Setup ---
# Use an image with distinct textures or objects (e.g., a book cover or building)
image_path = 'boochan.png' 

# --- Step 2: Load Image ---
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not load image from {image_path}")
else:
    # --- Step 3: Convert to Grayscale ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Step 4: Create SIFT Object ---
    # This initializes the algorithm parameters
    sift = cv2.SIFT_create()

    # --- Step 5: Detect and Compute ---
    # kp: List of Keypoint objects (contains x,y coords, size, angle)
    # des: Numpy array of shape (Number of Keypoints, 128)
    kp, des = sift.detectAndCompute(gray, None)

    print(f"Number of keypoints detected: {len(kp)}")
    
    # --- Step 6: Draw Keypoints ---
    # We use the flag DRAW_RICH_KEYPOINTS to draw circles with size and orientation
    # instead of just dots. This visualizes the "Scale-Invariant" nature.
    img_sift = cv2.drawKeypoints(
        gray, 
        kp, 
        img, 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # --- Step 7: Display ---
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB))
    plt.title(f'SIFT Detection ({len(kp)} keypoints)')
    plt.axis('off')
    plt.show()

    # --- Step 8: Save Output ---
    cv2.imwrite('sift_output.jpg', img_sift)
    print("SIFT result saved.")