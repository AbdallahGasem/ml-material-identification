#                                                       بسم الله الرحمن الرحيم                                                 //
#  program: data_loader.py 
#  Description: loading the data, return the exctracted features along to with the labels
#  Author:  Abdallah Gasem
#  Date: 17-12-2025
#  Version: 1.0
# ----------------------------------------------------------------------------------------------------------------------------- //

import numpy as np
import os
import cv2
from .features.extractor import extract_features
import joblib


MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
CACHED_FILE = os.path.join(MODEL_DIR, 'features_cached.npz')


def load_resources(MODEL_PATH, SCALER_PATH, ENCODER_PATH):
    """
    Loads the ML model, the scaler, and the label encoder.
    Handles errors if files are missing.
    """
    print("Loading AI resources...")
    try:
        # TODO: Load the model using joblib
        model = joblib.load(MODEL_PATH)
        
        # TODO: Load the scaler using joblib
        scaler = joblib.load(SCALER_PATH)
        
        # TODO: Load the encoder using joblib
        encoder = joblib.load(ENCODER_PATH)
        
        return model, scaler, encoder
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: {e}")
        print("Run 'train_svm.py' first to generate these files.")
        exit()


def load_dataset(data_path, model_in_action=False):
    """
    Traverses the folder structure, reads images, and extracts features.
    
    Args:
        data_path: Path to the root data folder.
        model_in_action: deals with test data without labels
        
    Returns:
        X (np.array): Matrix of feature vectors (Shape: N_images x 8638)
        y (np.array): List of string labels (e.g., ['Glass', 'Metal', ...])
    """
    
    # Lists to hold our data
    X = []
    y = []
    
    # ---------------------------------------------------------
    # STEP 1: LOAD AND EXTRACT
    # ---------------------------------------------------------
    # Loop through every subfolder (Class Name) in data_path.
    # For each image file:
    #   1. Read it with cv2.imread.
    #   2. Check if it loaded correctly (not None).
    #   3. Pass it to extract_features(image).
    #   4. Append the result to X.
    #   5. Append the folder name (label) to y.
    
    # Hint: Use os.listdir(data_path) to get class names.
    # Hint: Use os.path.join() to build paths safely.
    
    # TODO: Implement the loop.
    
    if model_in_action:  
        if not os.path.isdir(data_path):
            print(f"Warning class path: {data_path} not found, Coudn't extract the data!")
            print(f"make sure that the data path: {data_path} contain the image directly without subdirs representing classes! other wise set model_in_action=False")
            return

        print(f'processing images in : {data_path}...')
        
        image_files = [file for file in os.listdir(data_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for image in image_files:
            image_path = os.path.join(data_path, image)
            img = cv2.imread(image_path)
            
            if img is None:
                print(f"Warning: Could not read {image_path}, Skipping...")
                continue
            
            features = extract_features(img)
            X.append(features)
        return X
        
    CLASSES = os.listdir(data_path)
    for class_dir in CLASSES:
        class_path = os.path.join(data_path, class_dir)
        
        if not os.path.isdir(class_path):
            print(f"Warning class path: {class_path} not found")
            continue
        
        print(f'processing class: {class_dir}...')
        
        image_files = [file for file in os.listdir(class_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for image in image_files:
            image_path = os.path.join(class_path, image)
            img = cv2.imread(image_path)
            
            if img is None:
                print(f"Warning: Could not read {image_path}")
                continue
            
            features = extract_features(img)
            X.append(features)
            y.append(class_dir)
    
    # Convert lists to numpy arrays for Scikit-Learn
    return np.array(X), np.array(y)
    

# the time to load the data this way is So long that is we need to save the results to a file
# csv is bad as we have about 8200 columns that is as per the ai recommendation numpy format .npz is better!
def get_data_with_cache(data_path, ignore_cache=False):
    """
    Checks if features are already saved. 
    If yes -> Load from disk (Fast).
    If no -> Extract from images and save to disk (Slow first time).
    
    Args:
        data_path: Path to the root data folder.
        
    Returns:
        X (np.array): Matrix of feature vectors (Shape: N_images x 8638)
        y (np.array): List of string labels (e.g., ['Glass', 'Metal', ...])
    
    """
    
    # check the cached file first
    if os.path.exists(CACHED_FILE) and not ignore_cache:
        print(f'loading data from the cached file: {CACHED_FILE}')
        data = np.load(CACHED_FILE)
        X = data['X']
        y = data['y']
        return X, y

    print(f'CACHE FILE NOT FOUND: {CACHED_FILE}!')
    print(f'Loading data from scratch...')
    X, y = load_dataset(data_path)
    
    print(f'Saving Loaded data at {CACHED_FILE}')
    np.savez_compressed(CACHED_FILE, X=X, y=y)
        
    return X, y
    
    
if __name__ == '__main__':
    get_data_with_cache('data_augmented')