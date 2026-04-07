import pandas as pd
import numpy as np
import os

def generate_disease_data(n_samples=2000):
    np.random.seed(42)
    
    # Contextual features
    age = np.random.randint(20, 80, n_samples)
    bmi = np.random.normal(25, 5, n_samples)
    smoke = np.random.randint(0, 2, n_samples)
    alcohol = np.random.randint(0, 2, n_samples)
    physical_activity = np.random.randint(0, 3, n_samples) # 0: low, 1: med, 2: high
    diet_quality = np.random.randint(0, 3, n_samples) # 0: poor, 1: avg, 2: good
    sleep_hours = np.random.normal(7, 1.5, n_samples)
    stress_level = np.random.randint(0, 10, n_samples)
    work_environment = np.random.randint(0, 3, n_samples) # 0: sedentary, 1: mixed, 2: active
    family_history_diabetes = np.random.randint(0, 2, n_samples)
    family_history_heart = np.random.randint(0, 2, n_samples)
    
    # Medical features
    blood_pressure_systolic = np.random.normal(120, 15, n_samples)
    blood_pressure_diastolic = np.random.normal(80, 10, n_samples)
    glucose = np.random.normal(100, 30, n_samples)
    cholesterol = np.random.normal(200, 40, n_samples)
    creatinine = np.random.normal(1.0, 0.4, n_samples)
    alt_liver = np.random.normal(25, 15, n_samples)
    
    # Calculate Probabilities for Each Disease (Simplistic but logical rule-based for synthetic data)
    prob_diabetes = (glucose > 125) * 0.4 + (bmi > 30) * 0.2 + (family_history_diabetes == 1) * 0.2 + (age > 50) * 0.1 + (physical_activity == 0) * 0.1
    prob_heart = (blood_pressure_systolic > 140) * 0.3 + (cholesterol > 240) * 0.2 + (smoke == 1) * 0.2 + (family_history_heart == 1) * 0.15 + (age > 55) * 0.1 + (stress_level > 7) * 0.05
    prob_kidney = (creatinine > 1.2) * 0.4 + (blood_pressure_systolic > 140) * 0.2 + (glucose > 140) * 0.2 + (age > 60) * 0.2
    prob_liver = (alt_liver > 40) * 0.5 + (alcohol == 1) * 0.3 + (bmi > 30) * 0.1 + (diet_quality == 0) * 0.1

    # Add noise
    prob_diabetes += np.random.normal(0, 0.1, n_samples)
    prob_heart += np.random.normal(0, 0.1, n_samples)
    prob_kidney += np.random.normal(0, 0.1, n_samples)
    prob_liver += np.random.normal(0, 0.1, n_samples)

    # Classify based on thresholds
    diabetes = (prob_diabetes > 0.5).astype(int)
    heart = (prob_heart > 0.5).astype(int)
    kidney = (prob_kidney > 0.5).astype(int)
    liver = (prob_liver > 0.5).astype(int)

    df = pd.DataFrame({
        'age': age, 'bmi': bmi, 'smoke': smoke, 'alcohol': alcohol, 
        'physical_activity': physical_activity, 'diet_quality': diet_quality,
        'sleep_hours': sleep_hours, 'stress_level': stress_level,
        'work_environment': work_environment,
        'family_history_diabetes': family_history_diabetes,
        'family_history_heart': family_history_heart,
        'blood_pressure_systolic': blood_pressure_systolic,
        'blood_pressure_diastolic': blood_pressure_diastolic,
        'glucose': glucose, 'cholesterol': cholesterol,
        'creatinine': creatinine, 'alt_liver': alt_liver,
        'diabetes': diabetes, 'heart_disease': heart,
        'kidney_disease': kidney, 'liver_disease': liver
    })

    os.makedirs('dataset', exist_ok=True)
    df.to_csv('dataset/multi_disease_data.csv', index=False)
    print(f"Generated synthetic data with {n_samples} samples.")
    print("Disease distributions:")
    print(df[['diabetes', 'heart_disease', 'kidney_disease', 'liver_disease']].mean())

if __name__ == '__main__':
    generate_disease_data()
