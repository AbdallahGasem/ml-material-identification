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


#--- Configuration ---#
MODEL_PATH = "models/svm/best_svm_model.joblib"
SCALER_PATH = "models/svm/scaler.joblib"
ENCODER_PATH = "models/svm/label_encoder.joblib"
DATA_PATH = 'test_data'

CONFIDENCE_THRESHOLD = 0.40 # 40%


def run_test(dataFilePath, bestModelPath):
    #load data
    X = load_dataset(data_path=DATA_PATH, model_in_action=True)
    
    # load the model, scaler & encoder
    model, scaler, _ = load_resources(bestModelPath, SCALER_PATH, ENCODER_PATH) # return the IDs
    
    # predict the data and return the array of predictions
    X_scaled = scaler.transform(X)
    
    # apply the unkown class id 6!
    probs = model.predict_proba(X_scaled)
    
    final_predictions = []
    
    for prob_vector in probs:
        best_class_id = np.argmax(prob_vector)
        confidence = prob_vector[best_class_id]
        
        if confidence < CONFIDENCE_THRESHOLD:
            final_predictions.append(6) # Append 6
        else:
            final_predictions.append(best_class_id)    # Append 0-5
            
    # Return the list of integers
    return final_predictions


if __name__ == '__main__':
    preds = run_test(DATA_PATH, MODEL_PATH)
    print(preds)