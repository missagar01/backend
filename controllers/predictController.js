const Prediction = require('../models/Prediction');
const { exec } = require('child_process');
const path = require('path');

// ═══════════════════════════════════════════════════════════════
// PREDICT — Execute AI Ensemble & Store Research-Grade Results
// ═══════════════════════════════════════════════════════════════
exports.predict = async (req, res) => {
  try {
    const features = req.body;
    // UPDATED PATH: Expecting ml_models inside backend/ for Render
    const scriptPath = path.join(__dirname, '..', 'ml_models', 'inference.py');

    // Fetch previous assessment for temporal analysis
    let previousDiseaseData = null;
    try {
      const lastPrediction = await Prediction.findOne({ userId: req.user._id })
        .sort({ createdAt: -1 })
        .lean();

      if (lastPrediction && lastPrediction.diseases) {
        previousDiseaseData = {};
        for (const [disease, data] of Object.entries(lastPrediction.diseases)) {
          previousDiseaseData[disease] = {
            risk_probability: data.risk_probability || 0
          };
        }
      }
    } catch (e) {
      // If no previous data, temporal analysis will be null
    }

    // Build payload with features + previous assessment for temporal intelligence
    const mlPayload = { ...features };
    if (previousDiseaseData) {
      mlPayload._previous_assessment = previousDiseaseData;
    }

    const inputJson = JSON.stringify(mlPayload).replace(/"/g, '\\"');

    exec(`py "${scriptPath}" "${inputJson}"`, { maxBuffer: 1024 * 1024 * 10 }, async (error, stdout, stderr) => {
      if (error) {
        console.error('Python execution error:', error);
        return res.status(500).json({ detail: 'ML Model Error', error: error.message });
      }

      try {
        const result = JSON.parse(stdout);
        if (result.error) throw new Error(result.error);

        const { overall_assessment, diseases, trend_analysis, model_comparison, meta } = result;

        // Save structured research-grade data to MongoDB
        const predictionRecord = await Prediction.create({
          userId: req.user._id,
          input_features: features,
          overall_assessment,
          diseases,
          trend_analysis,
          model_comparison,
          meta
        });

        // Return rich API response
        res.status(200).json({
          id: predictionRecord._id,
          timestamp: predictionRecord.createdAt,
          input_features: features,
          overall_assessment,
          diseases,
          trend_analysis,
          model_comparison,
          meta
        });
      } catch (parseError) {
        console.error('JSON Parse Error:', parseError.message);
        console.error('Python stdout:', stdout);
        console.error('Python stderr:', stderr);
        res.status(500).json({ detail: 'Error processing ML output', error: parseError.message });
      }
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ detail: 'Server Error', error: err.message });
  }
};

// ═══════════════════════════════════════════════════════════════
// GET HISTORY — Rich longitudinal data for dashboard + trends
// ═══════════════════════════════════════════════════════════════
exports.getHistory = async (req, res) => {
  try {
    const history = await Prediction.find({ userId: req.user._id })
      .sort({ createdAt: -1 })
      .lean();

    const formattedHistory = history.map(h => ({
      id: h._id,
      timestamp: h.createdAt,
      input_features: h.input_features || {},
      overall_assessment: h.overall_assessment || {},
      diseases: h.diseases || {},
      trend_analysis: h.trend_analysis || null,
      predictions: {
        diabetes: h.diseases?.diabetes?.risk_level === 'High' ? 1 : 0,
        heart_disease: h.diseases?.heart_disease?.risk_level === 'High' ? 1 : 0,
        kidney_disease: h.diseases?.kidney_disease?.risk_level === 'High' ? 1 : 0,
        liver_disease: h.diseases?.liver_disease?.risk_level === 'High' ? 1 : 0,
      }
    }));

    res.status(200).json(formattedHistory);
  } catch (err) {
    console.error(err);
    res.status(500).json({ detail: 'History Retrieval Error' });
  }
};

// ═══════════════════════════════════════════════════════════════
// GET MODEL METRICS — For Research Analytics page
// ═══════════════════════════════════════════════════════════════
exports.getModelMetrics = async (req, res) => {
  try {
    const metricsPath = path.join(__dirname, '..', '..', 'ml_models', 'saved_models', 'metrics.json');
    const fs = require('fs');

    if (fs.existsSync(metricsPath)) {
      const metrics = JSON.parse(fs.readFileSync(metricsPath, 'utf-8'));
      res.status(200).json(metrics);
    } else {
      res.status(404).json({ detail: 'Metrics not found. Train models first.' });
    }
  } catch (err) {
    res.status(500).json({ detail: 'Error loading metrics' });
  }
};
