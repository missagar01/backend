import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import json

def train_models():
    data_path = 'dataset/multi_disease_data.csv'
    if not os.path.exists(data_path):
        from data_generator import generate_disease_data
        generate_disease_data()
    
    df = pd.read_csv(data_path)
    
    features = [
        'age', 'bmi', 'smoke', 'alcohol', 'physical_activity', 'diet_quality', 
        'sleep_hours', 'stress_level', 'work_environment', 
        'family_history_diabetes', 'family_history_heart', 
        'blood_pressure_systolic', 'blood_pressure_diastolic', 
        'glucose', 'cholesterol', 'creatinine', 'alt_liver'
    ]
    
    targets = ['diabetes', 'heart_disease', 'kidney_disease', 'liver_disease']
    
    X = df[features]
    
    metrics = {}
    os.makedirs('saved_models', exist_ok=True)
    
    for target in targets:
        print(f"\n--- Training models for {target} ---")
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize Base Models
        lr = LogisticRegression(max_iter=1000)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        knn = KNeighborsClassifier(n_neighbors=5)
        
        # Ensemble Model (Soft Voting to get probabilities for explainability)
        ensemble = VotingClassifier(
            estimators=[('lr', lr), ('rf', rf), ('xgb', xgb), ('knn', knn)],
            voting='soft'
        )
        
        models = {'LogisticRegression': lr, 'RandomForest': rf, 'XGBoost': xgb, 'KNN': knn, 'Ensemble': ensemble}
        target_metrics = {}
        
        best_model_name = ""
        best_f1 = 0
        best_model = None
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            target_metrics[name] = {
                'Accuracy': float(acc),
                'Precision': float(prec),
                'Recall': float(rec),
                'F1-Score': float(f1)
            }
            print(f"{name} -> Accuracy: {acc:.4f}, F1: {f1:.4f}")
            
            if name == 'Ensemble':
                # Always save the ensemble for our context-aware system
                joblib.dump(model, f'saved_models/{target}_ensemble.pkl')
                
        metrics[target] = target_metrics
    
    # Save training metrics
    with open('saved_models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("\nTraining completed. Models and metrics saved.")

if __name__ == '__main__':
    train_models()
