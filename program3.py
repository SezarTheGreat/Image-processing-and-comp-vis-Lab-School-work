import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'boochan.png' 
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not load image from {image_path}")
    # Creating a dummy image for demonstration if file is missing
    img = np.zeros((600, 600, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (500, 500), (255, 255, 255), -1)
    cv2.putText(img, "Sample", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

# Convert to RGB for display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- Step 2: Select 4 Points from Original Image ---
# These are the (x, y) coordinates of the 4 corners of the object in the image.
# Order: Top-Left, Top-Right, Bottom-Left, Bottom-Right
# NOTE: In a real app, you would click these. Here, we approximate for the demo.
rows, cols, ch = img.shape
pts1 = np.float32([
    [100, 100],       # Top-Left point in original image
    [cols-100, 100],  # Top-Right point
    [50, rows-50],    # Bottom-Left (slanted)
    [cols-50, rows-50]# Bottom-Right (slanted)
])

# --- Step 3: Select 4 Destination Points ---
# Where do we want these points to go? 
# We want to "flatten" them into a perfect rectangle (e.g., 400x300 size)
width, height = 400, 300
pts2 = np.float32([
    [0, 0],           # Top-Left
    [width, 0],       # Top-Right
    [0, height],      # Bottom-Left
    [width, height]   # Bottom-Right
])

# --- Step 4: Compute Transformation Matrix ---
# getPerspectiveTransform solves the math to find the 3x3 Homography matrix
matrix = cv2.getPerspectiveTransform(pts1, pts2)

# --- Step 5: Apply Warp ---
# warpPerspective applies the matrix to every pixel
output = cv2.warpPerspective(img, matrix, (width, height))

# --- Step 6: Show and Save ---
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image (with marked points)")
# Draw the points on the original image for visualization
for point in pts1:
    plt.plot(point[0], point[1], 'ro') # Red dots

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Warped Perspective")

plt.show()

cv2.imwrite('warped_output.jpg', output)
print("Transformation complete. Saved as 'warped_output.jpg'")