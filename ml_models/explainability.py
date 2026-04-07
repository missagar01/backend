import os
import joblib
import numpy as np

def explain_prediction(features_dict, predictions_dict):
    """
    Given contextual features and disease predictions, generate an explanation.
    This simulates SHAP/LIME output by providing a human-readable reason based on feature thresholds.
    """
    reasons = {}
    
    # Feature thresholds mapping to risk
    high_risks = {
        'glucose': (125, 'high blood sugar'),
        'bmi': (30, 'high BMI (obesity level)'),
        'smoke': (1, 'smoking habits'),
        'blood_pressure_systolic': (140, 'high blood pressure'),
        'cholesterol': (240, 'high cholesterol'),
        'creatinine': (1.2, 'elevated creatinine levels'),
        'alt_liver': (40, 'elevated liver enzymes (ALT)'),
        'alcohol': (1, 'alcohol consumption'),
        'age': (55, 'advanced age risk factor'),
        'stress_level': (7, 'high stress level'),
        'family_history_diabetes': (1, 'family history of diabetes'),
        'family_history_heart': (1, 'family history of heart disease')
    }
    
    for disease, has_risk in predictions_dict.items():
        if has_risk:
            disease_reasons = []
            if disease == 'diabetes':
                for feat in ['glucose', 'bmi', 'family_history_diabetes', 'age']:
                    if features_dict.get(feat, 0) >= high_risks[feat][0]:
                        disease_reasons.append(high_risks[feat][1])
            elif disease == 'heart_disease':
                for feat in ['blood_pressure_systolic', 'cholesterol', 'smoke', 'family_history_heart', 'age', 'stress_level']:
                    if features_dict.get(feat, 0) >= high_risks[feat][0]:
                        disease_reasons.append(high_risks[feat][1])
            elif disease == 'kidney_disease':
                for feat in ['creatinine', 'blood_pressure_systolic', 'glucose', 'age']:
                    if features_dict.get(feat, 0) >= high_risks[feat][0]:
                        disease_reasons.append(high_risks[feat][1])
            elif disease == 'liver_disease':
                for feat in ['alt_liver', 'alcohol', 'bmi']:
                    if features_dict.get(feat, 0) >= high_risks[feat][0]:
                        disease_reasons.append(high_risks[feat][1])
            
            if disease_reasons:
                reasons[disease] = f"High {disease.replace('_', ' ')} risk due to {', '.join(disease_reasons)}."
            else:
                reasons[disease] = f"Elevated {disease.replace('_', ' ')} risk detected from complex interacting factors (ensemble prediction)."
                
    return reasons
