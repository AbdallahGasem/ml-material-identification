#                                                       بسم الله الرحمن الرحيم                                                 //
#  program: train_knn.py 
#  Description: 
#  Author:  Abdallah Gasem
#  Date: 18-12-2025
#  Version: 1.0
# ----------------------------------------------------------------------------------------------------------------------------- //


import os
import joblib
from data_loader import get_data_with_cache
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from data_loader import get_data_with_cache 


#---Config---#
MODEL_DIR = 'models/knn'
os.makedirs(MODEL_DIR, exist_ok=True)
DATA_DIR = 'data_augmented'


def train_KNN():
    print("Starting k-NN Training Pipeline...")
    
    # ---------------------------------------------------------
    # STEP 1: LOAD DATA
    # ---------------------------------------------------------
    # TODO: Load X and y using your cached loader.
    X, y = get_data_with_cache(DATA_DIR)
    
    
    # ---------------------------------------------------------
    # STEP 2: PREPROCESSING (ENCODING & SPLITTING)
    # ---------------------------------------------------------
    # TODO: Encode the string labels (y) into integers.
    
    # TODO: Split the data into Train and Test sets.
    # Requirement: Ensure the class distribution is preserved (stratified).
    # Requirement: Use a consistent random_seed.
    Label_encoder = LabelEncoder()
    y_encoded = Label_encoder.fit_transform(y)
    
    x_train, x_test, y_train, y_test = train_test_split(X, y_encoded, random_state=1234, stratify=y_encoded)
    joblib.dump(Label_encoder, os.path.join(MODEL_DIR, 'label_encoder.joblib'))
    
    # ---------------------------------------------------------
    # STEP 3: FEATURE SCALING (CRITICAL FOR k-NN)
    # ---------------------------------------------------------
    # k-NN calculates the Euclidean distance between points.
    # If one feature has a range [0, 1] and another [0, 1000], the second one dominates.
    
    # TODO: Initialize and fit a Scaler on the Training data.
    # TODO: Transform both Train and Test data.
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))

    
    # -----------------------------------------------------------------------------------------------------------------
    # # ---------------------------------------------------------
    # # STEP 4: TRAIN KNN
    # # --------------------------------------------------------    
    # # You must test three specific things:
    # #   1. 'n_neighbors': Test a range of ODD numbers (e.g., 3, 5, 7, 9, 11).
    # #      (Why odd? To avoid ties in voting).
    # #   2. 'weights': Test both 'uniform' (democracy) and 'distance' (closer neighbors have more influence).
    # #   3. 'metric': Test 'euclidean' (L2) and 'manhattan' (L1).
    
    # # TODO: Initialize the KNN Classifier.
    
    # print('KNN Model Training...')
    
    # knn_model = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')
    # knn_model.fit(x_train_scaled, y_train)
    
    # # ---------------------------------------------------------
    # # STEP 5: EVALUATE AND SAVE
    # # ---------------------------------------------------------
    # print('Model Testing...')
    
    
    # # TODO: Predict on the Test Set.
    # preds = knn_model.predict(x_test_scaled)
    # -----------------------------------------------------------------------------------------------------------------
        
    # ---------------------------------------------------------
    # STEP 4: TRAIN KNN (WITH TUNING)
    # ---------------------------------------------------------
    print('Tuning k-NN Hyperparameters...')
    
    # Define what we want to test
    param_grid = [
        {
            'n_neighbors': [3, 5, 7, 9],       # Odd numbers to avoid ties
            'weights': ['uniform', 'distance'], # Democracy vs Distance-weighted
            'metric': ['euclidean', 'manhattan'] # Straight line vs Grid distance
        }
    ]
    
    knn = KNeighborsClassifier()
    
    # Run Grid Search (cv=3 means it checks 3 times for validation)
    clf = GridSearchCV(knn, param_grid, verbose=1, cv=3, n_jobs=-1)
    
    clf.fit(x_train_scaled, y_train)
    
    print(f"Best Parameters Found: {clf.best_params_}")
    
    # ---------------------------------------------------------
    # STEP 5: EVALUATE AND SAVE
    # ---------------------------------------------------------
    print('Model Testing...')
    
    # Get the winner
    best_knn_model = clf.best_estimator_
    
    preds = best_knn_model.predict(x_test_scaled)
    acc = accuracy_score(y_test, preds)    
        
        
    # TODO: Calculate and print the Accuracy Score.
    acc =accuracy_score(y_test, preds)
    print(f'KNN Model Accuracy: {acc}')
    # TODO: Print the full Classification Report with correct class names.
    print('Detailed Report:')
    print(classification_report(y_test, preds, target_names=Label_encoder.classes_))
    # TODO: Save ONLY the best k-NN model to 'models/knn_model.joblib'.
    joblib.dump(best_knn_model, os.path.join(MODEL_DIR, 'best_knn_model.joblib'))
    
    print("Pipeline Complete.")





if __name__ == "__main__":
    train_KNN()
    

'''

Starting k-NN Training Pipeline...
loading data from the cached file: models\features_cached.npz
Tuning k-NN Hyperparameters...
Fitting 3 folds for each of 16 candidates, totalling 48 fits
Best Parameters Found: {'metric': 'manhattan', 'n_neighbors': 3, 'weights': 'distance'}
Model Testing...
KNN Model Accuracy: 0.7331499312242091
Detailed Report:
              precision    recall  f1-score   support

   cardboard       0.80      0.79      0.79       122
       glass       0.72      0.73      0.72       121
       metal       0.78      0.63      0.70       122
       paper       0.79      0.73      0.76       118
     plastic       0.56      0.68      0.61       120
       trash       0.80      0.85      0.82       124

    accuracy                           0.73       727
   macro avg       0.74      0.73      0.73       727
weighted avg       0.74      0.73      0.73       727

Pipeline Complete.

'''