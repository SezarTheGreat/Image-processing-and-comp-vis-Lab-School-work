import cv2
import numpy as np

# ----------------------------
# 1. Load the image
# ----------------------------
img = cv2.imread("boochan.png")
img = cv2.resize(img, (400, 400))   # resizing for easy processing

# Convert to a 2D array of pixels
pixel_vals = img.reshape((-1, 3))
pixel_vals = np.float32(pixel_vals)

# ----------------------------
# 2. Define criteria & clusters
# ----------------------------
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            20, 0.001)

k = 4   # number of clusters
_, labels, centers = cv2.kmeans(pixel_vals, k, None,
                                criteria, 10,
                                cv2.KMEANS_RANDOM_CENTERS)

# ----------------------------
# 3. Convert back to image form
# ----------------------------
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape(img.shape)

# ----------------------------
# 4. Save and show result
# ----------------------------
cv2.imwrite("segmented_output.jpg", segmented_image)
print("Segmented image saved as segmented_output.jpg")