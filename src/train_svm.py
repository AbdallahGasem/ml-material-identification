#                                                       بسم الله الرحمن الرحيم                                                 //
#  program: train_svm.py 
#  Description: 
#  Author:  Abdallah Gasem
#  Date: 17-12-2025
#  Version: 1.0
# ----------------------------------------------------------------------------------------------------------------------------- //


def train():
    print("Starting Training Pipeline...")
    
    # ---------------------------------------------------------
    # STEP 2: PREPARE DATA
    # ---------------------------------------------------------
    # TODO: Call load_dataset().
    
    
    # ---------------------------------------------------------
    # STEP 3: ENCODE LABELS
    # ---------------------------------------------------------
    # SVM does not understand strings like "Metal". It needs numbers (0, 1, 2...).
    # Use LabelEncoder to convert y (strings) to y_encoded (numbers).
    
    # TODO: Fit and transform the labels.
    # Hint: le = LabelEncoder()
    
    # CRITICAL: Save the encoder! 
    # The real-time app needs to know that 0 = "Cardboard".
    # TODO: joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))
    
    
    # ---------------------------------------------------------
    # STEP 4: SPLIT TRAIN / TEST
    # ---------------------------------------------------------
    # We need to hide some data to test if the model is actually learning or just memorizing.
    # Standard split: 80% Training, 20% Testing.
    
    # TODO: Split X and y.
    # Hint: Use train_test_split.
    # Hint: Use stratify=y to ensure both Train and Test have equal amounts of "Trash" vs "Glass".
    
    
    # ---------------------------------------------------------
    # STEP 5: FEATURE SCALING (THE MOST IMPORTANT STEP)
    # ---------------------------------------------------------
    # Your HOG features might range from 0.0 to 0.4.
    # Your Color features might range from 0 to 255 (if not normalized correctly).
    # SVM calculates "Distance". Large numbers dominate small numbers.
    # StandardScaler forces all features to have Mean=0 and Variance=1.
    
    # TODO: Initialize StandardScaler.
    # TODO: Fit it on TRAINING data, then transform TRAINING data.
    # TODO: ONLY transform TEST data (Do not fit on test data, that is "Data Leakage").
    
    # CRITICAL: Save the scaler!
    # The real-time app needs to scale the camera input exactly the same way.
    # TODO: joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    
    
    # ---------------------------------------------------------
    # STEP 6: TRAIN SVM
    # ---------------------------------------------------------
    # TODO: Initialize the SVC (Support Vector Classifier).
    # parameters to consider:
    #   kernel='linear': Good for high-dimensional data (8000+ features).
    #   C=1.0: Default regularization.
    #   probability=True: MANDATORY for the "Unknown" class logic later.
    
    # TODO: Fit the model on X_train_scaled and y_train.
    
    
    # ---------------------------------------------------------
    # STEP 7: EVALUATE AND SAVE
    # ---------------------------------------------------------
    # TODO: Predict on X_test_scaled.
    # TODO: Print Accuracy Score.
    # TODO: Print Classification Report (Precision/Recall per class).
    
    # TODO: Save the trained model.
    # joblib.dump(svm_model, os.path.join(MODEL_DIR, "best_model.joblib"))
    
    print("Pipeline Complete.")

if __name__ == "__main__":
    train()