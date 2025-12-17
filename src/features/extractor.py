#                                                       بسم الله الرحمن الرحيم                                                 //
#  program: extractor.py 
#  Description: feature Descriptors Combinator
#  Author:  Abdallah Gasem
#  Date: 17-12-2025
#  Version: 1.0
# ----------------------------------------------------------------------------------------------------------------------------- //

from hsv import extract_color_histogram
from hog import extract_hog_features
from lbp import extract_lbp_features
import numpy as np
import cv2

def extract_features(image):
    hog_features = extract_hog_features(image)
    hsv_features = extract_color_histogram(image)
    lbp_features = extract_lbp_features(image)
    final_features_vector = np.hstack([hog_features, hsv_features, lbp_features])
    
    return final_features_vector 
    
    

if __name__ == '__main__':
    # loading and testing an image!
    img = cv2.imread("data_augmented/metal/0ac64aca-c63c-41a1-90ce-000211985691.jpg")
    if img is not None:
        print(len(extract_features(img)))
    else:
        print("invalid path")