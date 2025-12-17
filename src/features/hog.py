#                                                       بسم الله الرحمن الرحيم                                                 //
#  program: hog.py 
#  Description: building the HOG feature Deciptor
#  Author:  Abdallah Gasem
#  Date: 17-12-2025
#  Version: 1.0
# ----------------------------------------------------------------------------------------------------------------------------- //


import cv2
import numpy as np
from skimage.feature import hog

def extract_hog_features(image):
    """
    Extracts HOG (Shape) features from an image.
    
    Args:
        image: A numpy array representing the input image (BGR).
        
    Returns:
        np.array: A 1D flattened feature vector representing the shape structure.
    """
    
    # ---------------------------------------------------------
    # STEP 1: RESIZING (Crucial for HOG)
    # ---------------------------------------------------------
    # HOG works on a fixed "window" of pixels. If your images have 
    # different sizes, the feature vector length will change (bad for SVM).
    # TODO: Resize the image to a fixed size (e.g., 64x128 or 128x128).
    # Hint: Use cv2.resize().
    resized_image = cv2.resize(image, (128, 128))
    
    # ---------------------------------------------------------
    # STEP 2: GRAYSCALE CONVERSION
    # ---------------------------------------------------------
    # HOG relies on light intensity changes, not color.
    # TODO: Convert the resized image to Grayscale.
    # Hint: Use cv2.cvtColor with COLOR_BGR2GRAY.
    gs_resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # ---------------------------------------------------------
    # STEP 3: COMPUTE HOG DESCRIPTORS
    # ---------------------------------------------------------
    # This is the heavy lifting. You can do this manually using Sobel filters
    # or use skimage.feature.hog (recommended for this project's deadline).
    
    # IF USING SKIMAGE.FEATURE.HOG, YOU MUST DEFINE:
    # 1. orientations: Number of direction bins (usually 9).
    #    (e.g., vertical, horizontal, diagonal, etc.)
    # 2. pixels_per_cell: Size of the smallest unit (e.g., 8x8 pixels).
    #    The histogram is calculated for each "cell".
    # 3. cells_per_block: How many cells to group for Normalization (e.g., 2x2).
    #    This makes the feature robust to shadows (local contrast).
    
    # TODO: Call the HOG function.
    # Hint: Ensure 'feature_vector=True' (or similar) to get a 1D array back.
    # Hint: Enable 'transform_sqrt' to reduce the effect of shadows/glare.
    hog_feature_vector = hog(gs_resized_image, pixels_per_cell=(8,8), cells_per_block=(2,2), 
                             orientations=9, transform_sqrt=True, feature_vector=True)
    
    return np.array(hog_feature_vector)


if __name__ == '__main__':
    # loading and testing an image!
    img = cv2.imread("data_augmented/metal/0ac64aca-c63c-41a1-90ce-000211985691.jpg")
    if img is not None:
        print(extract_hog_features(img).shape)
    else:
        print("invalid path")