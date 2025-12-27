#                                                       بسم الله الرحمن الرحيم                                                //
#  program: app.py 
#  Description: Realtime app deployment
#  Author: Abdallah Gasem
#  Date: 16-12-2025
#  Version: 1.0
# ----------------------------------------------------------------------------------------------------------------------------- //


import cv2
import numpy as np
import joblib
from features.extractor import extract_features
from data_loader import load_resources

# --- CONFIGURATION ---
# Paths to your trained "brains"
MODEL_PATH = "models/svm/best_svm_model.joblib"
SCALER_PATH = "models/svm/scaler.joblib"
ENCODER_PATH = "models/svm/label_encoder.joblib"

CONFIDENCE_THRESHOLD = 0.60     # rejection rule??????


def main():
    # ---------------------------------------------------------
    # STEP 1: INITIALIZATION
    # ---------------------------------------------------------
    # TODO: Call load_resources() to get your model/scaler/encoder.
    model, scaler, encoder = load_resources(MODEL_PATH, SCALER_PATH, ENCODER_PATH)
    
    # TODO: Open the webcam using cv2.VideoCapture(0).
    stream = cv2.VideoCapture(0)
    
    # Check if opened successfully.
    if not stream.isOpened():
        print("ERROR: couldn't open the camera, system will shutdown...")
        return
    
    print("--- System Ready. Press 'q' to quit. ---")
    

    while True:
        # ---------------------------------------------------------
        # STEP 2: CAPTURE AND SETUP
        # ---------------------------------------------------------
        
        # Define the "Scan Zone" (A square box in the center).
        # We process ONLY what is inside this box to avoid background noise.
        # height, width = frame.shape[:2]
        # box_size = 250
        # Calculate x1, y1, x2, y2 for the center.
        
        # TODO: Read a frame from the camera.
        # TODO: Crop the frame to get the Region of Interest (ROI).
        # roi = frame[y1:y2, x1:x2]
        
        state, frame = stream.read()
        
        if not state:
            print('ERROR: camera disconnected or the frame is not readable!')
            break
        
        h, w, _ = frame.shape
        box_size = 200
        
        x1 = (w//2)-(box_size//2) 
        y1 = (h//2)-(box_size//2) 
        x2 = x1 + box_size
        y2 = y1 + box_size
        region_of_interest = frame[y1:y2, x1:x2]
        
        display_text = "Analyzing..."
        color = (0, 165, 255) # Orange (Default)
        
        
        # ---------------------------------------------------------
        # STEP 3: PREDICT
        # ---------------------------------------------------------
        try:
            # 1. EXTRACT: Get features from the ROI (using your extractor).
            features_vector = extract_features(region_of_interest)
            
            # 2. SHAPE: Reshape features to (1, -1) because the model expects a batch.
            features_vector = features_vector.reshape(1,-1)
            
            # 3. SCALE: Transform features using the loaded scaler.
            features_scaled = scaler.transform(features_vector)
            
            # 4. PREDICT: Get probabilities using model.predict_proba().
            # Example result: [0.1, 0.8, 0.1] -> Class 1 is 80% likely.
            probs = model.predict_proba(features_scaled)[0]
            best_class_idx = np.argmax(probs)
            confidence = probs[best_class_idx]
            
            # 5. FILTER: 
            # If max(probability) > CONFIDENCE_THRESHOLD:
            #    Get the class name using encoder.inverse_transform().
            #    Set display_text = "Plastic (85%)"
            #    Set color = (0, 255, 0) # Green
            
            if confidence > CONFIDENCE_THRESHOLD:
                class_name = encoder.inverse_transform([best_class_idx])[0]
                display_text = f"{class_name} ({confidence*100:.1f}%)"
                color = (0,255,0)
            else:
                display_text = 'Unknown!'
                color = (0, 165, 255)
            
        except Exception as e:
            display_text = 'ERROR!'
            color = (255,0,0)
            print(e)

        # ---------------------------------------------------------
        # STEP 4: VISUALIZATION
        # ---------------------------------------------------------
        # TODO: Draw the Green/Orange Rectangle on the original frame.
        
        # Draw the box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw text background (so it's readable)
        cv2.putText(frame, display_text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Show the frame
        cv2.imshow('Material Identification System', frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # ---------------------------------------------------------
        # STEP 5: EXIT LOGIC
        # ---------------------------------------------------------
        # Check if 'q' is pressed to break the loop.
    
    stream.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
