import cv2
import matplotlib.pyplot as plt

# --- Configuration ---
IMAGE1_PATH = 'boochan.png'
IMAGE2_PATH = 'boochan2.png'
BINS = 32
COMPARISON_METHOD = cv2.HISTCMP_CORREL

# --- Helper Function ---
def get_histogram(image_path, bins):
    """Load image, convert to grayscale, and calculate normalized histogram."""
    img_color = cv2.imread(image_path)
    if img_color is None:
        return None, None
    
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    # Calculate Histogram
    hist = cv2.calcHist([img_gray], [0], None, [bins], [0, 256])
    
    # Normalize and flatten
    hist = cv2.normalize(hist, hist).flatten()
    return hist, cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

# --- Main Program ---
if __name__ == "__main__":
    # Step 1: Read and convert both images to grayscale
    hist1, img1_rgb = get_histogram(IMAGE1_PATH, BINS)
    if hist1 is None:
        print(f"Error: Could not load '{IMAGE1_PATH}'")
        exit()
    
    hist2, img2_rgb = get_histogram(IMAGE2_PATH, BINS)
    if hist2 is None:
        print(f"Error: Could not load '{IMAGE2_PATH}'")
        exit()
    
    # Step 2: Compare histograms
    similarity_score = cv2.compareHist(hist1, hist2, COMPARISON_METHOD)
    
    # Step 3: Display results
    print(f"Image 1: {IMAGE1_PATH}")
    print(f"Image 2: {IMAGE2_PATH}")
    print(f"Similarity Score (Correlation): {similarity_score:.6f}")
    
    # Step 4: Show both images side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Histogram Similarity Comparison\nScore: {similarity_score:.6f}", fontsize=14)
    
    axes[0].imshow(img1_rgb)
    axes[0].set_title("boochan.png")
    axes[0].axis('off')
    
    axes[1].imshow(img2_rgb)
    axes[1].set_title("boochan2.png")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()