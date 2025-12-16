#                                                       بسم الله الرحمن الرحيم                                                //
#  program: augment.py 
#  Description: 
#  Author:  Abdallah Gasem
#  Date: 16-12-2025
#  Version: 1.0
# ----------------------------------------------------------------------------------------------------------------------------- //

import cv2
import numpy as np
import os
import random
import shutil
from pathlib import Path

# --- CONFIGURATION ---

INPUT_DIR = Path('../data')
OUTPUT_DIR = Path('../data_augmented')
TARGET_COUNT = 500  # stated in the project that we need to make up upto 500 each! 


# Define the classes (excluding 'Unknown' if you want, but usually we augment it too)
CLASSES = ['glass', 'paper', 
           'cardboard', 'plastic', 
           'metal', 'trash']

# --- AUGMENTATION FUNCTIONS (The "How") ---
# Rotaion, noise, brightness, fliping

def augment_rotate(image):
    """Rotates image by a random angle between -25 and 25 degrees."""
    angle = random.uniform(-25, 25)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Perform rotation
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def augment_noise(image):
    """Adds random Gaussian noise (simulates bad camera sensor)."""
    row, col, ch = image.shape
    mean = 0
    var = 100 # Adjust variance for more/less noise
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    # Clip values to stay between 0-255
    return np.clip(noisy, 0, 255).astype(np.uint8)

def augment_brightness(image):
    """Randomly adjusts brightness (simulates different lighting)."""
    value = random.uniform(0.7, 1.3) # 70% to 130% brightness
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * value
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255 # Cap at 255
    hsv = np.array(hsv, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def augment_flip(image):
    """Horizontal flip."""
    return cv2.flip(image, 1)

# List of available augmentations
aug_options = [augment_noise, augment_brightness, augment_flip, augment_rotate]
# # --- MAIN PIPELINE ---

def augment():
    
    # prepare Output dir #
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)   # cleaning the existing dir
    OUTPUT_DIR.mkdir(parents=True)
    
    total_orginal_count = 0
    total_new = 0
    
    # defining paths for each class
    for class_dir in CLASSES:
        class_in_path = INPUT_DIR/class_dir    
        class_out_path = OUTPUT_DIR/class_dir
        
        if not class_in_path.exists():
            print(f'Folder{class_dir} not exists!')
            continue
        
        # get image files
        image_files = [file for file in os.listdir(class_in_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        current_class_count = len(image_files)
        total_orginal_count += current_class_count 
        
        print(f"processing{class_dir}: found {current_class_count} Images!") 
        
        # create output directory for this class
        class_out_path.mkdir(parents=True, exist_ok=True)
        
        # copying the orignal to the output folder
        for image in image_files:
            src = class_in_path / image
            destination = class_out_path / image
            shutil.copy(src,destination)
        
        # calculating how many do we need to generate
        needed = max(0, TARGET_COUNT-current_class_count)
        print(f"  -> Generating {needed} new images...")
        
        # generate loop
        generated_count = 0
        while(generated_count < needed):
            rand_image_name = random.choice(image_files)
            img_path = class_in_path / rand_image_name  # shouldn't it be class_out_path?? 
            img = cv2.imread(img_path)
            
            if img is None: continue
            
            # pick a random agmentation function
            aug_func = random.choice(aug_options)
            # save the augmented image
            aug_img_name = f'aug_{generated_count}_{rand_image_name}'
            cv2.imwrite(str(class_out_path/aug_img_name), aug_func(img))
            
            generated_count += 1
        total_new += generated_count

    print("-" * 30)
    print("Augmentation Completed.")
    print(f"Original Count: {total_orginal_count}")
    print(f"Total Augmented Images Created: {total_new}")
    print(f"New Dataset Size: {total_new + total_orginal_count}")
    
    
if __name__ == "__main__":
    augment()