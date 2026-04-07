const mongoose = require('mongoose');

const predictionSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
    index: true
  },

  // ═══════════════════════════════════════════
  // INPUT FEATURES (Clinical Data Intake)
  // ═══════════════════════════════════════════
  input_features: {
    age: Number,
    bmi: Number,
    smoke: Number,
    alcohol: Number,
    physical_activity: Number,
    diet_quality: Number,
    sleep_hours: Number,
    stress_level: Number,
    work_environment: Number,
    family_history_diabetes: Number,
    family_history_heart: Number,
    blood_pressure_systolic: Number,
    blood_pressure_diastolic: Number,
    glucose: Number,
    cholesterol: Number,
    creatinine: Number,
    alt_liver: Number
  },

  // ═══════════════════════════════════════════
  // OVERALL ASSESSMENT (AI Health Score)
  // ═══════════════════════════════════════════
  overall_assessment: {
    health_score: Number,
    total_risks_detected: Number,
    total_diseases_screened: Number,
    primary_concern: String,
    risk_classification: String,
    average_risk_probability: Number
  },

  // ═══════════════════════════════════════════
  // PER-DISEASE RESULTS (Rich structured data)
  // ═══════════════════════════════════════════
  diseases: {
    diabetes: { type: mongoose.Schema.Types.Mixed, default: {} },
    heart_disease: { type: mongoose.Schema.Types.Mixed, default: {} },
    kidney_disease: { type: mongoose.Schema.Types.Mixed, default: {} },
    liver_disease: { type: mongoose.Schema.Types.Mixed, default: {} }
  },

  // ═══════════════════════════════════════════
  // TEMPORAL INTELLIGENCE (Trend Analysis)
  // ═══════════════════════════════════════════
  trend_analysis: {
    type: mongoose.Schema.Types.Mixed,
    default: null
  },

  // ═══════════════════════════════════════════
  // MODEL COMPARISON METRICS
  // ═══════════════════════════════════════════
  model_comparison: {
    type: mongoose.Schema.Types.Mixed,
    default: {}
  },

  // ═══════════════════════════════════════════
  // SYSTEM METADATA
  // ═══════════════════════════════════════════
  meta: {
    algorithm: String,
    framework: String,
    explainability: String,
    confidence_method: String,
    version: String,
    total_models: Number,
    ensemble_strategy: String
  }

}, {
  timestamps: true
});

// Compound index for efficient user history queries
predictionSchema.index({ userId: 1, createdAt: -1 });

module.exports = mongoose.model('Prediction', predictionSchema);
