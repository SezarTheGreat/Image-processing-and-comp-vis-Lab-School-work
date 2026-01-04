import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
INPUT_IMAGE_PATH = "boochan.png"
RESIZE_DIM = (800, 600)
K_CLUSTERS = 4 # Number of clusters for K-Means segmentation

# --- Helper Functions (Implementing your conceptual steps) ---

def read_and_resize(path, dim):
    """Reads the image and resizes it."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Error: Image not found at {path}")
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

def reconstruct_kmeans_image(labels, centers, original_shape):
    """Reconstructs the image from K-Means cluster labels and centers."""
    centers = np.uint8(centers)
    # Map the labels back to the corresponding cluster center colors
    reconstructed_pixels = centers[labels.flatten()]
    return reconstructed_pixels.reshape(original_shape)

def get_rough_rectangle(img, margin=50):
    """Defines a rough rectangle for GrabCut initiation."""
    h, w = img.shape[:2]
    return (margin, margin, w - 2 * margin, h - 2 * margin)


# --- Main Segmentation Pipeline ---

try:
    # Read and Resize Image
    img = read_and_resize(INPUT_IMAGE_PATH, RESIZE_DIM)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Storage for results
    results = {}

    # =================================================================
    ## 1. K-Means Segmentation (Clustering by Color)
    # =================================================================
    print("Running K-Means Segmentation...")
    
    # Reshape the image into a 2D array of pixels (N_pixels, 3_color_channels)
    pixels = rgb.reshape((-1, 3))
    pixels = np.float32(pixels)
    
    # Define criteria for K-Means (stop if 100 iterations or epsilon 0.2 is reached)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # Apply K-Means
    # compact: sum of squared distances from each point to its assigned center
    ret, labels, centers = cv2.kmeans(pixels, K_CLUSTERS, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Reconstruct Image
    kmeans_result = reconstruct_kmeans_image(labels, centers, rgb.shape)
    results['K-Means Result'] = kmeans_result
    
    # =================================================================
    ## 2. GrabCut Segmentation (User-Guided Masking)
    # =================================================================
    print("Running GrabCut Segmentation...")
    
    # Initialize the mask and the models
    mask = np.zeros(img.shape[:2], np.uint8) # 0: Background
    bgdModel = np.zeros((1, 65), np.float64) # Background model
    fgdModel = np.zeros((1, 65), np.float64) # Foreground model
    
    # Define the initial rough rectangle (margin=50)
    rect = get_rough_rectangle(img, margin=50)
    
    # Apply GrabCut (uses energy minimization to refine the mask)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    
    # Create the final mask: 1 & 3 are Foreground, 0 & 2 are Background
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Apply mask to the original image
    grabcut_result = rgb * mask[:, :, np.newaxis]
    results['GrabCut Result'] = grabcut_result
    
    # =================================================================
    ## 3. Watershed Segmentation (Structure-Based Flooding)
    # =================================================================
    print("Running Watershed Segmentation...")
    
    # Thresholding (Otsu's Threshold)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological Operations
    # Remove small white noise (opened)
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Determine background (dilate the opened image)
    background = cv2.dilate(opened, kernel, iterations=3)
    
    # Determine sure foreground (Distance Transform)
    # Higher value means further from background (center of object)
    dist_transform = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    
    # Threshold the distance transform to get sure foreground
    ret, foreground = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)
    foreground = np.uint8(foreground)
    
    # Find unknown region (The boundary area where background and foreground meet)
    unknown = cv2.subtract(background, foreground)
    
    # Prepare Markers (Find connected components in the foreground)
    ret, markers = cv2.connectedComponents(foreground)
    
    # Label the unknown region with 0 (Watershed requires this)
    # Label the background region (which is 255 in unknown) with the total markers + 1
    markers[unknown == 255] = 0 
    
    # Apply Watershed
    markers = cv2.watershed(img, markers)
    
    # Color the boundaries (where markers == -1) in red (0, 0, 255 in BGR)
    img[markers == -1] = [0, 0, 255] 
    watershed_result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results['Watershed Result'] = watershed_result
    
    # =================================================================
    ## Display All Results
    # =================================================================
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot K-Means Result
    axes[0].imshow(results['K-Means Result'])
    axes[0].set_title(f"K-Means (K={K_CLUSTERS})")
    axes[0].axis('off')

    # Plot GrabCut Result
    axes[1].imshow(results['GrabCut Result'])
    axes[1].set_title("GrabCut (Semi-Automatic)")
    axes[1].axis('off')

    # Plot Watershed Result
    axes[2].imshow(results['Watershed Result'])
    axes[2].set_title("Watershed (Structure-Based)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"\n--- EXECUTION FAILED ---")
    print(f"Please place an image file named '{INPUT_IMAGE_PATH}' in the script directory to run the segmentation.")
    print(f"---")
except Exception as e:
    print(f"\n--- An unexpected error occurred during execution: {e} ---")
