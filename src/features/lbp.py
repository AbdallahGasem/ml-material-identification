
#                                                       بسم الله الرحمن الرحيم                                                 //
#  program: lbp.py 
#  Description: building the lbp feature Descriptor
#  Author:  Abdallah Gasem
#  Date: 17-12-2025
#  Version: 1.0
# ----------------------------------------------------------------------------------------------------------------------------- //


import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def extract_lbp_features(image, num_points=24, radius=8):
    """
    Extracts texture features using LBP.
    
    Args:
        image: Input image (BGR).
        num_points: Number of neighbors to check around the center pixel.
        radius: How far away the neighbors are (pixel distance).
    
    Returns:
        np.array: A 1D histogram of the texture patterns.
    """
    
    # ---------------------------------------------------------
    # STEP 1: GRAYSCALE
    # ---------------------------------------------------------
    # LBP works on intensity comparisons.
    # TODO: Convert to grayscale.
    gs_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    # ---------------------------------------------------------
    # STEP 2: COMPUTE LBP IMAGE
    # ---------------------------------------------------------
    # This step transforms the image from "Colors" to "Texture Codes".
    # Every pixel becomes a number representing its local pattern.
    
    # TODO: Run the LBP algorithm.
    # Hint: Use skimage.feature.local_binary_pattern
    # Hint: Use method='uniform' to reduce noise and keep only strong patterns.
    lbp = local_binary_pattern(gs_image, P=num_points, R=radius, method='uniform')
    
    # ---------------------------------------------------------
    # STEP 3: BUILD THE HISTOGRAM
    # ---------------------------------------------------------
    # We don't feed the LBP image itself to the SVM (that's too much data).
    # We feed the *distribution* of textures.
    # e.g., "This image is 40% 'rough pattern' and 60% 'smooth pattern'."
    
    # TODO: Create a histogram of the LBP output.
    # Hint: np.histogram
    # Hint: The number of bins depends on 'num_points'.
    #       For 'uniform' LBP, max bins = num_points + 2.
    
    hist, _ = np.histogram(lbp.ravel(), bins=num_points+2, range=(0, num_points+2))
    
    
    # ---------------------------------------------------------
    # STEP 4: NORMALIZE
    # ---------------------------------------------------------
    # Same as Color Histogram. We need probabilities, not raw counts.
    
    # TODO: Normalize the histogram so it sums to 1.
    # Hint: hist = hist.astype("float") / hist.sum()
    hist = hist.astype("float")
    hist /= (hist.sum() + 0.0000001)  # AI recommendation to avoid dividing by Zero
    
    # ---------------------------------------------------------
    # STEP 5: RETURN
    # ---------------------------------------------------------
    # TODO: Return the feature vector.
    
    return hist



if __name__ == '__main__':
    # loading and testing an image!
    img = cv2.imread("data_augmented/metal/0ac64aca-c63c-41a1-90ce-000211985691.jpg")
    if img is not None:
        print(extract_lbp_features(img).shape)
    else:
        print("invalid path")