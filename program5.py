import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 0. Setup and Helper Functions ---
# Define a placeholder width/height for the intrinsic matrix calculation
# You would get these from your actual image size
WIDTH = 800
HEIGHT = 600

def read_image(path, grayscale=False):
    """Loads an image and converts it to the required format."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Error: Image not found at {path}")
    if grayscale:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def points_from(keypoints, matches, type_str):
    """Extracts matched keypoint coordinates for calculation."""
    # 'query' corresponds to kp1, 'train' corresponds to kp2
    indices = [m.queryIdx if type_str == "query" else m.trainIdx for m in matches]
    return np.float32([keypoints[i].pt for i in indices])

def camera_matrix(fx, fy, cx, cy):
    """Creates the 3x3 Intrinsic Camera Matrix (K)."""
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)

def projection_matrix(R, t):
    """Creates the 3x4 Projection Matrix [R | t]."""
    # R is 3x3, t is 3x1. Concatenate horizontally.
    return np.hstack((R, t))


# --- 1. Load Images ---
try:
    # Use placeholder names for the stereo pair
    img_gray1 = read_image("boochan.png", grayscale=True)
    img_color1 = read_image("boochan.png", grayscale=False) 
    img_gray2 = read_image("boochan2.png", grayscale=True)
except FileNotFoundError as e:
    print(e)
    print("--- Using placeholder image data for demonstration ---")
    img_gray1 = np.ones((HEIGHT, WIDTH), dtype=np.uint8) * 128
    img_color1 = cv2.cvtColor(img_gray1, cv2.COLOR_GRAY2BGR)
    img_gray2 = np.ones((HEIGHT, WIDTH), dtype=np.uint8) * 128
    
    # NOTE: Since no valid images exist, the 3D calculation will fail 
    # but the script structure remains correct.
    
    # We exit the main processing logic if files are missing
    # In a real scenario, you'd ensure the files exist.
    # For this demo, we'll continue with the dummy data to show the flow.
    # If using dummy data, the 3D points will be nonsensical.


# --- 2. ORB Feature Detection and Matching ---
# ORB (Oriented FAST and Rotated BRIEF) is a fast, efficient alternative to SIFT
orb = cv2.ORB_create(nfeatures=4000)
kp1, des1 = orb.detectAndCompute(img_gray1, None)
kp2, des2 = orb.detectAndCompute(img_gray2, None)

# --- 3. Feature Matching (BFMatcher) ---
# BFMatcher (Brute-Force Matcher) using HAMMING distance for ORB descriptors
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(des1, des2)

# --- 4. Take Best Matches ---
# Sort by distance (smaller distance is better match) and take the top 1000
matches = sorted(matches, key=lambda x: x.distance)
num_best_matches = min(1000, len(matches))
matches = matches[:num_best_matches]
print(f"Found and kept {len(matches)} best matches.")

# --- 5. Extract Matched Points ---
pts1 = points_from(kp1, matches, "query")
pts2 = points_from(kp2, matches, "train")

# --- 6. Define Camera Intrinsic Matrix (K) ---
# Assuming a camera with a focal length of 1000 pixels
K = camera_matrix(fx=1000, fy=1000, cx=WIDTH/2, cy=HEIGHT/2)
print("\nIntrinsic Matrix K:\n", K)


# --- 7. Compute Essential Matrix and Camera Pose ---
# E = find_essential_mat(pts1, pts2, K)
# RANSAC method is used to robustly find E and filter outliers (mask)
E, mask = cv2.findEssentialMat(
    pts1, pts2, K, 
    method=cv2.RANSAC, 
    prob=0.999, 
    threshold=1.0 # Pixel difference threshold
)

# R, t = recover_pose(E, pts1, pts2, K)
# Recover the rotation (R) and translation (t) from the Essential Matrix
# t is the baseline vector (distance between the two camera centers)
_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

print("\nEstimated Rotation Matrix (R):\n", R)
print("Estimated Translation Vector (t):\n", t.T)


# --- 8. Triangulate 3D Points ---
# P0 = Projection Matrix of Camera 1 (Identity Rotation, Zero Translation)
P0 = projection_matrix(np.eye(3), np.zeros((3, 1))) 
# P1 = Projection Matrix of Camera 2 (Relative R and t)
P1 = projection_matrix(R, t)

# pts4D = triangulate(P0, P1, pts1, pts2)
# Triangulate points using the two projection matrices and the 2D matches
pts4D_homogeneous = cv2.triangulatePoints(P0, P1, pts1.T, pts2.T)


# --- 9. Convert Homogeneous to 3D ---
# Homogeneous coordinates are 4D (x, y, z, w). To get 3D, divide by w.
points_3D = pts4D_homogeneous / pts4D_homogeneous[3]
points_3D = points_3D[:3].T # Keep x, y, z and transpose back to (N, 3) shape

print(f"\nSuccessfully triangulated {points_3D.shape[0]} points.")
print("Sample 3D Points (X, Y, Z coordinates):")
print(points_3D[0:5]) # Print the first 5 points

# --- 10. Visualization (Optional but recommended) ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D points
ax.scatter(points_3D[:, 0], points_3D[:, 1], points_3D[:, 2], s=5, c='blue')

# Plot Camera 1 (at origin)
ax.scatter(0, 0, 0, c='red', marker='o', s=100, label='Camera 1')

# Plot Camera 2 (at translation vector t)
ax.scatter(t[0], t[1], t[2], c='green', marker='^', s=100, label='Camera 2')

ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_zlabel('Z (Depth)')
ax.set_title("3D Point Cloud Reconstruction")
ax.legend()
plt.show()