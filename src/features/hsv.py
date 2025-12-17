#                                                       بسم الله الرحمن الرحيم                                                 //
#  program: hsv.py 
#  Description: building the hsv feature Descriptor
#  Author:  Abdallah Gasem
#  Date: 17-12-2025
#  Version: 1.0
# ----------------------------------------------------------------------------------------------------------------------------- //

import cv2
import numpy as np

def extract_color_histogram(image, bins=(8, 8, 8)):
    """
    Computes the color histogram distribution.
    
    Args:
        image: Input image (BGR).
        bins: Tuple, how many 'buckets' to divide each channel into.
              (8, 8, 8) means we reduce the color space to 8x8x8 = 512 total combinations.
    
    Returns:
        np.array: A 1D normalized histogram.
    """
    
    # ---------------------------------------------------------
    # STEP 1: CONVERT TO HSV
    # ---------------------------------------------------------
    # TODO: Transform the image from BGR to HSV color space.
    # Hint: cv2.cvtColor
    
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # as cv reads images in bgr format!
    
    
    # ---------------------------------------------------------
    # STEP 2: COMPUTE THE HISTOGRAM
    # ---------------------------------------------------------
    # We need to count pixels falling into each bin.
    # Inputs needed for the function:
    #   - The image (wrapped in a list like [image])
    #   - Channels to use: [0, 1, 2] for (H, S, V)
    #   - Mask: None (we want the whole image)
    #   - HistSize: The 'bins' tuple you passed in.
    #   - Ranges: [0, 180, 0, 256, 0, 256] (Hue is 0-179 in OpenCV, others 0-255).
    
    # TODO: Calculate the histogram.
    # Hint: cv2.calcHist
    hist = cv2.calcHist([hsv_img], histSize=bins, mask=None, channels=[0,1,2], ranges=[0, 180, 0, 256, 0, 256])
    
    
    # ---------------------------------------------------------
    # STEP 3: NORMALIZE (CRITICAL STEP)
    # ---------------------------------------------------------
    # If Image A has 10 pixels and Image B has 10,000 pixels, their histograms
    # will look different even if they are identical content.
    # Normalization makes the sum of the histogram = 1 (or fits it to a scale).
    
    # TODO: Normalize the histogram.
    # Hint: cv2.normalize with NORM_L2 or MinMax.
    hsv_feature_vector = cv2.normalize(hist, None, norm_type=cv2.NORM_L2)
    
    # ---------------------------------------------------------
    # STEP 4: FLATTEN
    # ---------------------------------------------------------
    # The histogram usually comes out as a 3D block (8x8x8).
    # Machine Learning models need a flat list (1D array).
    
    # TODO: Flatten the array.
    # Hint: .flatten()
    return np.array(hsv_feature_vector).flatten()


if __name__ == '__main__':
    # loading and testing an image!
    img = cv2.imread("data_augmented/metal/0ac64aca-c63c-41a1-90ce-000211985691.jpg")
    if img is not None:
        print(extract_color_histogram(img).shape)
    else:
        print("invalid path")