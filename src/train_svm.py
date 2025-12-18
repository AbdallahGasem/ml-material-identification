#                                                       بسم الله الرحمن الرحيم                                                 //
#  program: train_svm.py 
#  Description: 
#  Author:  Abdallah Gasem
#  Date: 18-12-2025
#  Version: 1.0
# ----------------------------------------------------------------------------------------------------------------------------- //
import os

from sklearn.metrics import accuracy_score, classification_report
from data_loader import get_data_with_cache
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC


#---Config---#
MODEL_DIR = 'models/svm'
os.makedirs(MODEL_DIR, exist_ok=True)
DATA_DIR = 'data_augmented'



def train_SVM():
    print("Starting Training Pipeline...")
    
    # ---------------------------------------------------------
    # STEP 1: PREPARE DATA
    # ---------------------------------------------------------
    # TODO: Call load_dataset().
    X, y = get_data_with_cache(DATA_DIR)
    
    
    
    # ---------------------------------------------------------
    # STEP 2: ENCODE LABELS
    # ---------------------------------------------------------
    # SVM does not understand strings like "Metal". It needs numbers (0, 1, 2...).
    # Use LabelEncoder to convert y (strings) to y_encoded (numbers).
    
    # TODO: Fit and transform the labels.
    # Hint: le = LabelEncoder()
    
    # CRITICAL: Save the encoder! 
    # The real-time app needs to know that 0 = "Cardboard".
    # TODO: joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))
    print("Encoding labels...")
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, 'label_encoder.joblib'))
    
    
    
    # ---------------------------------------------------------
    # STEP 3: SPLIT TRAIN / TEST
    # ---------------------------------------------------------
    # We need to hide some data to test if the model is actually learning or just memorizing.
    # Standard split: 80% Training, 20% Testing.
    
    # TODO: Split X and y.
    # Hint: Use train_test_split.
    # Hint: Use stratify=y to ensure both Train and Test have equal amounts of "Trash" vs "Glass".
    print("Spliting the data...")

    x_train, x_test, y_train, y_test = train_test_split(X, y_encoded, random_state=1234, stratify=y_encoded)
    
    
    
    # ---------------------------------------------------------
    # STEP 4: FEATURE SCALING (THE MOST IMPORTANT STEP)
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
    print("Scalling the data...")
    
    standard_scaler = StandardScaler()
    x_train_scaled = standard_scaler.fit_transform(x_train)
    x_test_scaled = standard_scaler.transform(x_test)
    
    joblib.dump(standard_scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))
    
    
    
    # -------------------------------------------------------------------------------- 
    # # ---------------------------------------------------------
    # # STEP 5: TRAIN SVM
    # # ---------------------------------------------------------
    # # TODO: Initialize the SVC (Support Vector Classifier).
    # # parameters to consider:
    # #   kernel='linear': Good for high-dimensional data (8000+ features).
    # #   C=1.0: Default regularization.
    # #   probability=True: MANDATORY for the "Unknown" class logic later.
    
    # # TODO: Fit the model on X_train_scaled and y_train.
    # print('SVM model training...')
    
    # svm_model = SVC(kernel='linear', probability=True, random_state=1234, C=1.0)
    # svm_model.fit(X=x_train_scaled, y=y_train)  
    # -------------------------------------------------------------------------------- 
    
    # ---------------------------------------------------------
    # STEP 5: TRAIN SVM (WITH GRID SEARCH)
    # ---------------------------------------------------------
    print('Training SVM with Grid Search...')
    print('(This tries multiple settings to find the best accuracy)')

    # Define the parameter grid
    # C: Controls strictness (High C = stricter, Low C = smoother)
    # gamma: Controls how far a single data point's influence reaches
    # kernel: 'rbf' is usually superior for computer vision
    param_grid = [
        {'C': [1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    # Initialize Base Model
    svc = SVC(probability=True, random_state=1234)

    # Setup Grid Search
    # refit=True means "Train the best model on the whole dataset automatically"
    # verbose=2 prints progress
    clf = GridSearchCV(svc, param_grid, verbose=2, cv=3, n_jobs=-1)

    # Train
    clf.fit(x_train_scaled, y_train)

    print(f"Best Parameters Found: {clf.best_params_}")
    print(f"Best Cross-Validation Score: {clf.best_score_:.2f}")

    # The 'clf' object now behaves exactly like the best model
    svm_model = clf.best_estimator_
        
        
        
    # ---------------------------------------------------------
    # STEP 7: EVALUATE AND SAVE
    # ---------------------------------------------------------
    # TODO: Predict on X_test_scaled.
    # TODO: Print Accuracy Score.
    # TODO: Print Classification Report (Precision/Recall per class).
    
    # TODO: Save the trained model.
    # joblib.dump(svm_model, os.path.join(MODEL_DIR, "best_model.joblib"))
    print('Model Testing...')
    
    preds = svm_model.predict(x_test_scaled)
    acc = accuracy_score(y_test, preds)
    joblib.dump(svm_model, os.path.join(MODEL_DIR, 'best_svm_model.joblib'))
    
    
    print('-'*30)
    print(f'Model accuracy: {acc}')
    print('detailed report')
    print(classification_report(y_test, preds, target_names=label_encoder.classes_))
    
    print("Pipeline Complete.")



if __name__ == "__main__":
    train_SVM()
    

'''
(This tries multiple settings to find the best accuracy)
Fitting 3 folds for each of 6 candidates, totalling 18 fits
[CV] END ......................C=1, gamma=0.0001, kernel=rbf; total time= 5.1min
[CV] END .......................C=1, gamma=0.001, kernel=rbf; total time= 5.2min
[CV] END .......................C=1, gamma=0.001, kernel=rbf; total time= 5.2min
[CV] END .......................C=1, gamma=0.001, kernel=rbf; total time= 5.3min
[CV] END ......................C=1, gamma=0.0001, kernel=rbf; total time= 4.4min
[CV] END ......................C=1, gamma=0.0001, kernel=rbf; total time= 4.4min
[CV] END ......................C=10, gamma=0.001, kernel=rbf; total time= 4.5min
[CV] END ......................C=10, gamma=0.001, kernel=rbf; total time= 4.6min
[CV] END ......................C=10, gamma=0.001, kernel=rbf; total time= 4.5min
[CV] END .....................C=10, gamma=0.0001, kernel=rbf; total time= 4.4min
[CV] END .....................C=10, gamma=0.0001, kernel=rbf; total time= 4.4min
[CV] END .....................C=10, gamma=0.0001, kernel=rbf; total time= 4.4min
[CV] END .....................C=100, gamma=0.001, kernel=rbf; total time= 4.5min
[CV] END .....................C=100, gamma=0.001, kernel=rbf; total time= 4.6min
[CV] END ....................C=100, gamma=0.0001, kernel=rbf; total time= 4.4min
[CV] END .....................C=100, gamma=0.001, kernel=rbf; total time= 4.5min
[CV] END ....................C=100, gamma=0.0001, kernel=rbf; total time= 2.6min
[CV] END ....................C=100, gamma=0.0001, kernel=rbf; total time= 2.6min
Best Parameters Found: {'C': 10, 'gamma': 0.0001, 'kernel': 'rbf'}
Best Cross-Validation Score: 0.71
Model Testing...
------------------------------
Model accuracy: 0.7647867950481431
detailed report
              precision    recall  f1-score   support

   cardboard       0.87      0.85      0.86       122
       glass       0.67      0.70      0.69       121
       metal       0.70      0.72      0.71       122
       paper       0.75      0.79      0.77       118
     plastic       0.75      0.63      0.68       120
       trash       0.85      0.89      0.87       124

    accuracy                           0.76       727
   macro avg       0.76      0.76      0.76       727
weighted avg       0.76      0.76      0.76       727

Pipeline Complete.

'''