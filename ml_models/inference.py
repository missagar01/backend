import sys
import json
import os
import pandas as pd
import numpy as np
import joblib

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

FEATURE_NAMES = [
    'age', 'bmi', 'smoke', 'alcohol', 'physical_activity', 'diet_quality',
    'sleep_hours', 'stress_level', 'work_environment',
    'family_history_diabetes', 'family_history_heart',
    'blood_pressure_systolic', 'blood_pressure_diastolic',
    'glucose', 'cholesterol', 'creatinine', 'alt_liver'
]

TARGETS = ['diabetes', 'heart_disease', 'kidney_disease', 'liver_disease']
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')
MODEL_NAMES = ['Logistic Regression', 'Random Forest', 'XGBoost', 'KNN']

# =============================================================================
# 1. MATHEMATICALLY CORRECT CONFIDENCE FORMULA
# =============================================================================
# Confidence = w1*boundary_distance + w2*model_agreement + w3*(1-variance)
# Where:
#   boundary_distance = |ensemble_prob - 0.5| * 2  (range 0-1)
#   model_agreement   = 1 - (std_dev / max_possible_std)  (range 0-1)
#   variance_penalty  = 1 - np.var(probs) * 4  (normalized, range 0-1)
# Weights: w1=0.4, w2=0.35, w3=0.25

def compute_confidence(probs):
    """Research-grade confidence score using 3-factor weighted formula."""
    ensemble_prob = float(np.mean(probs))
    
    # Factor 1: Distance from decision boundary (0.5)
    boundary_distance = abs(ensemble_prob - 0.5) * 2.0
    
    # Factor 2: Model agreement (inverse of disagreement)
    std_dev = float(np.std(probs))
    max_std = 0.5  # max possible std when models are split 50-50
    agreement = 1.0 - min(std_dev / max_std, 1.0)
    
    # Factor 3: Low variance bonus
    variance = float(np.var(probs))
    variance_score = max(0, 1.0 - variance * 4.0)
    
    # Weighted combination
    confidence = 0.40 * boundary_distance + 0.35 * agreement + 0.25 * variance_score
    confidence = min(max(confidence, 0.0), 1.0)
    
    return {
        "score": round(confidence, 3),
        "boundary_distance": round(boundary_distance, 3),
        "model_agreement": round(agreement, 3),
        "variance_score": round(variance_score, 3),
        "formula": "0.4*boundary + 0.35*agreement + 0.25*variance_bonus"
    }


# =============================================================================
# 2. REAL FEATURE IMPORTANCE (SHAP-style via Tree-based Feature Importance)
# =============================================================================

def compute_feature_importance(ensemble_model, df, features_dict, disease):
    """
    Extract real feature importance from tree-based models in the ensemble.
    Uses sklearn's feature_importances_ for RF/XGB and coef_ for Logistic Regression.
    This is a genuine model-intrinsic explanation — not simulated.
    """
    importances = np.zeros(len(FEATURE_NAMES))
    count = 0
    
    if hasattr(ensemble_model, 'estimators_'):
        for estimator in ensemble_model.estimators_:
            if hasattr(estimator, 'feature_importances_'):
                # Random Forest, XGBoost
                importances += estimator.feature_importances_
                count += 1
            elif hasattr(estimator, 'coef_'):
                # Logistic Regression (use absolute coefficient values)
                importances += np.abs(estimator.coef_[0])
                count += 1
    
    if count > 0:
        importances = importances / count  # average across models
    
    # Normalize to sum to 1
    total = importances.sum()
    if total > 0:
        importances = importances / total
    
    # Build structured output
    feature_impacts = []
    for i, fname in enumerate(FEATURE_NAMES):
        val = features_dict.get(fname, 0)
        impact = float(importances[i])
        if impact > 0.02:  # Only include meaningful features
            feature_impacts.append({
                "feature": fname.replace('_', ' ').title(),
                "feature_key": fname,
                "importance": round(impact, 4),
                "value": val,
                "type": classify_feature_impact(fname, val, disease)
            })
    
    # Sort by importance descending
    feature_impacts.sort(key=lambda x: x['importance'], reverse=True)
    return feature_impacts


def classify_feature_impact(feature, value, disease):
    """Classify whether a feature value is a risk factor or protective factor."""
    risk_thresholds = {
        'glucose': 125, 'bmi': 30, 'smoke': 1, 'alcohol': 1,
        'blood_pressure_systolic': 140, 'blood_pressure_diastolic': 90,
        'cholesterol': 240, 'creatinine': 1.2, 'alt_liver': 40,
        'age': 55, 'stress_level': 7, 'sleep_hours': 5,
        'family_history_diabetes': 1, 'family_history_heart': 1,
        'work_environment': 2
    }
    
    protective_features = {'physical_activity', 'diet_quality', 'sleep_hours'}
    
    if feature in protective_features:
        if feature == 'sleep_hours':
            return "protective" if value >= 7 else "risk"
        return "protective" if value >= 1 else "risk"
    
    threshold = risk_thresholds.get(feature, None)
    if threshold is not None:
        return "risk" if value >= threshold else "protective"
    return "neutral"


# =============================================================================
# 3. NATURAL LANGUAGE EXPLANATION GENERATOR
# =============================================================================

def generate_explanation(disease, probability, top_features, features_dict):
    """Generate research-grade natural language explanation for each disease."""
    disease_label = disease.replace('_', ' ').title()
    risk_level = "elevated" if probability > 0.5 else "within normal range"
    
    # Get top 3 risk factors
    risk_factors = [f for f in top_features if f['type'] == 'risk'][:3]
    protective = [f for f in top_features if f['type'] == 'protective'][:2]
    
    # Build explanation
    explanation_parts = []
    
    if probability > 0.7:
        explanation_parts.append(f"{disease_label} risk is significantly elevated at {probability*100:.1f}%.")
    elif probability > 0.5:
        explanation_parts.append(f"{disease_label} risk is moderately elevated at {probability*100:.1f}%.")
    else:
        explanation_parts.append(f"{disease_label} risk is {risk_level} at {probability*100:.1f}%.")
    
    if risk_factors:
        factor_names = [f['feature'] for f in risk_factors]
        if len(factor_names) == 1:
            explanation_parts.append(f"Primary contributing factor: {factor_names[0]}.")
        else:
            explanation_parts.append(f"Key contributing factors include {', '.join(factor_names[:-1])} and {factor_names[-1]}.")
    
    if protective:
        prot_names = [f['feature'] for f in protective]
        explanation_parts.append(f"Protective factors: {', '.join(prot_names)}.")
    
    return ' '.join(explanation_parts)


# =============================================================================
# 4. PERSONALIZED RECOMMENDATION ENGINE
# =============================================================================

def generate_recommendations(disease, probability, features_dict, top_features):
    """Generate condition-specific, threshold-based, actionable recommendations."""
    recs = []
    
    if disease == 'diabetes':
        if features_dict.get('glucose', 0) > 125:
            recs.append({"priority": "high", "action": f"Reduce blood glucose below 125 mg/dL (current: {features_dict.get('glucose', 0)}).", "category": "lab"})
        if features_dict.get('bmi', 0) > 25:
            recs.append({"priority": "high", "action": f"Target BMI reduction to below 25 (current: {features_dict.get('bmi', 0)}).", "category": "lifestyle"})
        if features_dict.get('physical_activity', 0) < 2:
            recs.append({"priority": "medium", "action": "Increase physical activity — aim for 30 min brisk walking daily.", "category": "lifestyle"})
        if features_dict.get('diet_quality', 0) < 2:
            recs.append({"priority": "medium", "action": "Improve diet quality — reduce refined sugars, increase fiber intake.", "category": "diet"})
        if features_dict.get('family_history_diabetes', 0) == 1:
            recs.append({"priority": "info", "action": "Family history present — recommend annual HbA1c screening.", "category": "screening"})
    
    elif disease == 'heart_disease':
        if features_dict.get('blood_pressure_systolic', 0) > 140:
            recs.append({"priority": "high", "action": f"Control systolic BP below 140 mmHg (current: {features_dict.get('blood_pressure_systolic', 0)}).", "category": "lab"})
        if features_dict.get('cholesterol', 0) > 200:
            recs.append({"priority": "high", "action": f"Reduce total cholesterol below 200 mg/dL (current: {features_dict.get('cholesterol', 0)}).", "category": "lab"})
        if features_dict.get('smoke', 0) == 1:
            recs.append({"priority": "high", "action": "Smoking cessation is critical — consider nicotine replacement therapy.", "category": "lifestyle"})
        if features_dict.get('stress_level', 0) > 6:
            recs.append({"priority": "medium", "action": f"Reduce stress levels (current: {features_dict.get('stress_level', 0)}/10) — consider mindfulness or counseling.", "category": "mental"})
        if features_dict.get('physical_activity', 0) < 2:
            recs.append({"priority": "medium", "action": "Increase cardiovascular exercise — 150 min/week moderate-intensity.", "category": "lifestyle"})
    
    elif disease == 'kidney_disease':
        if features_dict.get('creatinine', 0) > 1.2:
            recs.append({"priority": "high", "action": f"Monitor serum creatinine closely (current: {features_dict.get('creatinine', 0)}). Consult nephrologist.", "category": "lab"})
        if features_dict.get('blood_pressure_systolic', 0) > 130:
            recs.append({"priority": "high", "action": "Strict BP control below 130/80 mmHg to protect renal function.", "category": "lab"})
        if features_dict.get('glucose', 0) > 125:
            recs.append({"priority": "medium", "action": "Control blood glucose — hyperglycemia accelerates kidney damage.", "category": "lab"})
        recs.append({"priority": "medium", "action": "Increase water intake to 2-3 liters/day. Limit sodium to <2g/day.", "category": "diet"})
    
    elif disease == 'liver_disease':
        if features_dict.get('alt_liver', 0) > 40:
            recs.append({"priority": "high", "action": f"Elevated ALT levels ({features_dict.get('alt_liver', 0)} U/L) — liver function tests recommended.", "category": "lab"})
        if features_dict.get('alcohol', 0) == 1:
            recs.append({"priority": "high", "action": "Reduce or eliminate alcohol consumption immediately.", "category": "lifestyle"})
        if features_dict.get('bmi', 0) > 30:
            recs.append({"priority": "medium", "action": "Weight reduction needed — fatty liver risk increases with BMI > 30.", "category": "lifestyle"})
        recs.append({"priority": "medium", "action": "Increase consumption of antioxidant-rich foods (green vegetables, berries).", "category": "diet"})
    
    # Add universal recommendation
    if probability > 0.5:
        recs.append({"priority": "high", "action": f"Schedule clinical consultation for {disease.replace('_', ' ')} evaluation.", "category": "clinical"})
    else:
        recs.append({"priority": "low", "action": "Continue annual health screening. Maintain current healthy habits.", "category": "screening"})
    
    return recs


# =============================================================================
# 5. TEMPORAL INTELLIGENCE — Compare with previous assessment
# =============================================================================

def compute_temporal_analysis(current_diseases, previous_assessment):
    """
    Compare current results with previous assessment.
    Returns trend direction, delta %, and anomaly flags.
    """
    if not previous_assessment:
        return None
    
    trends = {}
    for disease in TARGETS:
        curr_prob = current_diseases.get(disease, {}).get('risk_probability', 0)
        prev_prob = previous_assessment.get(disease, {}).get('risk_probability', 0)
        
        delta = curr_prob - prev_prob
        delta_pct = round(delta * 100, 1)
        
        if abs(delta) < 0.03:
            direction = "stable"
        elif delta > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        # Anomaly detection: sudden jump > 20%
        is_anomaly = abs(delta_pct) > 20
        
        trends[disease] = {
            "previous_probability": round(prev_prob, 3),
            "current_probability": round(curr_prob, 3),
            "delta": round(delta, 3),
            "delta_percentage": delta_pct,
            "direction": direction,
            "is_anomaly": is_anomaly
        }
    
    return trends


# =============================================================================
# MAIN INFERENCE ENGINE
# =============================================================================

def main():
    try:
        input_data = sys.argv[1]
        payload = json.loads(input_data)
        
        # Separate features from optional previous assessment
        features = {k: v for k, v in payload.items() if k in FEATURE_NAMES}
        previous_assessment = payload.get('_previous_assessment', None)
        
        df = pd.DataFrame([features])
        output_diseases = {}
        
        # Load metrics for model comparison
        metrics_path = os.path.join(MODELS_DIR, 'metrics.json')
        model_metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                model_metrics = json.load(f)
        
        for t in TARGETS:
            model_path = os.path.join(MODELS_DIR, f'{t}_ensemble.pkl')
            if os.path.exists(model_path):
                ensemble_model = joblib.load(model_path)
                
                # --- Per-model probabilities ---
                base_models = {}
                probs = []
                model_details = []
                
                if hasattr(ensemble_model, 'estimators_'):
                    for idx, estimator in enumerate(ensemble_model.estimators_):
                        try:
                            m_prob = float(estimator.predict_proba(df)[0][1])
                        except:
                            m_prob = float(estimator.predict(df)[0])
                        
                        name = MODEL_NAMES[idx]
                        base_models[name] = round(m_prob, 4)
                        probs.append(m_prob)
                        
                        # Determine each model's vote
                        model_details.append({
                            "name": name,
                            "probability": round(m_prob, 4),
                            "vote": "High Risk" if m_prob > 0.5 else "Low Risk",
                            "metrics": model_metrics.get(t, {}).get(name.replace(' ', ''), {})
                        })
                
                # --- Ensemble probability (soft voting) ---
                ensemble_prob = float(np.mean(probs)) if probs else 0.5
                final_pred = 1 if ensemble_prob > 0.5 else 0
                
                # --- Confidence Score (mathematical formula) ---
                confidence = compute_confidence(probs)
                
                # --- Model Agreement ---
                votes_high = sum(1 for p in probs if p > 0.5)
                votes_low = len(probs) - votes_high
                agreement_pct = round(max(votes_high, votes_low) / len(probs) * 100, 1) if probs else 0
                
                # --- Feature Importance (REAL) ---
                feature_impacts = compute_feature_importance(ensemble_model, df, features, t)
                top_risk = [f for f in feature_impacts if f['type'] == 'risk']
                top_protective = [f for f in feature_impacts if f['type'] == 'protective']
                
                # --- Natural Language Explanation ---
                explanation = generate_explanation(t, ensemble_prob, feature_impacts, features)
                
                # --- Personalized Recommendations ---
                recommendations = generate_recommendations(t, ensemble_prob, features, feature_impacts)
                
                output_diseases[t] = {
                    "risk_probability": round(ensemble_prob, 4),
                    "risk_level": "High" if final_pred == 1 else "Normal",
                    "confidence": confidence,
                    "model_outputs": base_models,
                    "model_details": model_details,
                    "ensemble_method": "Soft Voting (Probability Averaging)",
                    "agreement": {
                        "percentage": agreement_pct,
                        "votes_high": votes_high,
                        "votes_low": votes_low,
                        "consensus": "unanimous" if agreement_pct == 100 else "majority" if agreement_pct >= 75 else "split"
                    },
                    "top_factors": top_risk[:5],
                    "protective_factors": top_protective[:3],
                    "all_feature_importance": feature_impacts,
                    "explanation": explanation,
                    "recommendations": recommendations
                }
            else:
                output_diseases[t] = {
                    "risk_probability": 0.0,
                    "risk_level": "Unknown",
                    "confidence": {"score": 0},
                    "model_outputs": {},
                    "model_details": [],
                    "agreement": {"percentage": 0},
                    "top_factors": [],
                    "protective_factors": [],
                    "explanation": "Model not found.",
                    "recommendations": []
                }
        
        # --- Temporal Analysis ---
        trend_analysis = compute_temporal_analysis(output_diseases, previous_assessment)
        
        # --- AI Health Score ---
        all_probs = [d['risk_probability'] for d in output_diseases.values()]
        avg_risk = sum(all_probs) / len(all_probs) if all_probs else 0.5
        health_score = round((1 - avg_risk) * 100)
        
        # --- Risk summary ---
        total_risks = sum(1 for v in output_diseases.values() if v['risk_level'] == 'High')
        primary = max(output_diseases.items(), key=lambda x: x[1]['risk_probability'])[0] if output_diseases else "None"
        
        output = {
            "overall_assessment": {
                "health_score": health_score,
                "total_risks_detected": total_risks,
                "total_diseases_screened": 4,
                "primary_concern": primary,
                "risk_classification": "Critical" if total_risks >= 3 else "Elevated" if total_risks >= 1 else "Healthy",
                "average_risk_probability": round(avg_risk, 4)
            },
            "diseases": output_diseases,
            "trend_analysis": trend_analysis,
            "model_comparison": model_metrics,
            "meta": {
                "algorithm": "Hybrid Soft-Voting Ensemble (LR + RF + XGBoost + KNN)",
                "framework": "Scikit-Learn 1.x + XGBoost",
                "explainability": "Model-Intrinsic Feature Importance (RF Gini + XGB Gain + LR Coef)",
                "confidence_method": "3-Factor Weighted: Boundary Distance (0.4) + Agreement (0.35) + Variance (0.25)",
                "version": "2.0-research",
                "total_models": 4,
                "ensemble_strategy": "Soft Voting with Equal Weights"
            }
        }
        
        print(json.dumps(output))
        
    except Exception as e:
        import traceback
        print(json.dumps({"error": str(e), "traceback": traceback.format_exc()}))
        sys.exit(1)

if __name__ == '__main__':
    main()
