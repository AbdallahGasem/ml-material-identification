#                                                      بسم الله الرحمن الرحيم                                                 //
# program: test.py 
# Description: 
# Author:  Abdallah Gasem
# Date: 18-12-2025
# Version: 1.0
#----------------------------------------------------------------------------------------------------------------------------- //

'''
Dear students,
In the project, please submit a Python file called "test.py" with the deliverables
so that we can use it to determine your score on the hidden dataset.

This file should contain the necessary imports and a single function called "predict".
This function should take two parameters, namely the "dataFilePath" and the "bestModelPath",
and return a list of predictions. It should first load the images from the given folder path that directly contains
some images and load the model from its path. After that,
you can copy the code of your prediction pipeline (preprocessing, feature extraction, model inference, etc.)
and paste it inside this function, then make sure it runs without errors.

'''

from src.data_loader import load_dataset, load_resources
import numpy as np
import os
import pandas as pd


#--- Configuration ---#
MODEL_PATH = "models/svm/best_svm_model.joblib"
SCALER_PATH = "models/svm/scaler.joblib"
ENCODER_PATH = "models/svm/label_encoder.joblib"
DATA_PATH = 'sample_data/sample'

CONFIDENCE_THRESHOLD = 0.50 # 40%


def run_test(dataFilePath, bestModelPath):
    #load data
    X, pos_names = load_dataset(data_path=dataFilePath, model_in_action=True)
    
    # load the model, scaler & encoder
    model, scaler, encoder = load_resources(bestModelPath, SCALER_PATH, ENCODER_PATH) # return the IDs
    
    # predict the data and return the array of predictions
    X_scaled = scaler.transform(X)
    
    # apply the unkown class id 6!
    probs = model.predict_proba(X_scaled)
    
    final_predictions = []
    
    for prob_vector in probs:
        best_class_id = np.argmax(prob_vector)
        confidence = prob_vector[best_class_id]
        
        if confidence < CONFIDENCE_THRESHOLD:
            final_predictions.append("unknown") # Append 6: unkown
        else:
            class_name = encoder.inverse_transform([best_class_id])[0]
            final_predictions.append(class_name)
            
    # Return the list of integers
    df = pd.DataFrame({
        'image_name': pos_names,
        'classes': final_predictions
    })
    
    df.to_excel('predictions.xlsx', index=False)
    
    return final_predictions


if __name__ == '__main__':
    preds = run_test(DATA_PATH, MODEL_PATH)
    print(preds)