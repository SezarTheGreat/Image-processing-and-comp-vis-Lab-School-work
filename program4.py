import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Step 1 & 2: Imports ---
# (Done above)

# --- Step 3: Define a set of 3D world points (A Wireframe Cube) ---
def get_cube_points():
    # Define 8 corners of a cube centered at (0,0,0)
    points = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], # Front face
        [-1, -1, 1],  [1, -1, 1],  [1, 1, 1],  [-1, 1, 1]   # Back face
    ])
    return points.T # Transpose to shape (3, 8) for matrix math

# Define edges to draw lines between points (for visualization)
edges = [
    (0,1), (1,2), (2,3), (3,0), # Front face
    (4,5), (5,6), (6,7), (7,4), # Back face
    (0,4), (1,5), (2,6), (3,7)  # Connecting lines
]

# --- Step 4: Define Intrinsic Parameters ---
# f: Focal length (how "zoomed in" the lens is)
# cx, cy: Optical center (usually the center of the image)
f = 800 
cx, cy = 400, 300 

# Intrinsic Matrix (K)
K = np.array([
    [f, 0, cx],
    [0, f, cy],
    [0, 0, 1]
])

# --- Step 5: Define Extrinsic Parameters ---
# Rotation Matrix (R): Rotate the camera/object so we don't see it flat on
# Rotating 45 degrees around Y-axis and X-axis roughly
theta = np.radians(30)
# Rotation around Y-axis
R_y = np.array([
    [np.cos(theta), 0, np.sin(theta)],
    [0, 1, 0],
    [-np.sin(theta), 0, np.cos(theta)]
])
# Rotation around X-axis
R_x = np.array([
    [1, 0, 0],
    [0, np.cos(theta), -np.sin(theta)],
    [0, np.sin(theta), np.cos(theta)]
])

R = np.dot(R_x, R_y) # Combined rotation

# Translation Vector (T): Move the object AWAY from camera (positive Z)
# If Z is 0 or negative, the object is behind the camera!
T = np.array([[0], [0], [5]]) 

# --- Step 6: Convert World to Camera Coordinates ---
world_points = get_cube_points()
# Formula: P_cam = R * P_world + T
# We use broadcasting to add T to every point
camera_points = np.dot(R, world_points) + T

# --- Step 7: Apply Pinhole Projection ---
# Extract X, Y, Z rows
X_cam = camera_points[0, :]
Y_cam = camera_points[1, :]
Z_cam = camera_points[2, :]

# Formula: x = (f * X) / Z + cx
# We use the Intrinsic Matrix K for cleaner math usually, but let's do it manually
# to match your pseudocode logic:
u_coords = (f * X_cam) / Z_cam + cx
v_coords = (f * Y_cam) / Z_cam + cy

# --- Step 8: Plot the Projected 2D Points ---
plt.figure(figsize=(8, 6))
plt.title("3D to 2D Projection (Pinhole Model)")

# Plot the points (corners)
plt.scatter(u_coords, v_coords, color='red', label='Projected Points')

# Plot the lines (edges) to see the cube structure
for start_idx, end_idx in edges:
    p1 = (u_coords[start_idx], v_coords[end_idx])
    p2 = (u_coords[end_idx], v_coords[end_idx])
    plt.plot(
        [u_coords[start_idx], u_coords[end_idx]], 
        [v_coords[start_idx], v_coords[end_idx]], 
        'b-'
    )

# Set plot limits to simulate an image frame (e.g., 800x600)
plt.xlim(0, 800)
plt.ylim(600, 0) # Invert Y axis because images start from top-left
plt.legend()
plt.grid(True)
plt.show()

# --- Step 10: End ---
print("Projection complete.")