const Prediction = require('../models/Prediction');
const { spawn } = require('child_process');
const path = require('path');

// ═══════════════════════════════════════════════════════════════
// PREDICT — Execute AI Ensemble & Store Research-Grade Results
// ═══════════════════════════════════════════════════════════════
exports.predict = async (req, res) => {
  try {
    const features = req.body;
    // UPDATED PATH: ml_models is inside backend/
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
      console.log("No previous prediction found or error fetching it.");
    }

    // Build payload
    const mlPayload = { ...features };
    if (previousDiseaseData) {
      mlPayload._previous_assessment = previousDiseaseData;
    }

    const payloadString = JSON.stringify(mlPayload);

    // DYNAMIC COMMAND: Windows works best with 'py' (to avoid Microsoft Store alias), Render/Linux needs 'python3'
    const pythonCommand = process.platform === 'win32' ? 'py' : 'python3';
    
    console.log(`🚀 Spawning Python Engine (${pythonCommand}): ${scriptPath}`);
    const pythonProcess = spawn(pythonCommand, [scriptPath, payloadString]);

    let stdoutData = '';
    let stderrData = '';

    pythonProcess.stdout.on('data', (data) => {
      stdoutData += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      stderrData += data.toString();
      console.error(`🔴 Python Error: ${data.toString()}`);
    });

    pythonProcess.on('close', async (code) => {
      console.log(`🏁 Python process exited with code ${code}`);

      if (code !== 0) {
        return res.status(500).json({
          detail: 'ML Script Failed',
          error: stderrData || 'Unknown error'
        });
      }

      try {
        const result = JSON.parse(stdoutData);
        if (result.error) throw new Error(result.error);

        const { overall_assessment, diseases, trend_analysis, model_comparison, meta } = result;

        // Save to MongoDB
        const predictionRecord = await Prediction.create({
          userId: req.user._id,
          input_features: features,
          overall_assessment,
          diseases,
          trend_analysis,
          model_comparison,
          meta
        });

        res.status(200).json({
          id: predictionRecord._id,
          timestamp: predictionRecord.createdAt,
          ...result
        });
      } catch (parseError) {
        console.error('JSON Parse Error:', parseError.message);
        console.error('Raw Output:', stdoutData);
        res.status(500).json({
          detail: 'Error processing ML output',
          error: parseError.message,
          raw: stdoutData
        });
      }
    });

  } catch (err) {
    console.error('Server Error:', err);
    res.status(500).json({ detail: 'Server Error', error: err.message });
  }
};

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

exports.getModelMetrics = async (req, res) => {
  try {
    // UPDATED PATH: metrics is inside backend/ml_models/saved_models
    const metricsPath = path.join(__dirname, '..', 'ml_models', 'saved_models', 'metrics.json');
    const fs = require('fs');

    if (fs.existsSync(metricsPath)) {
      const metrics = JSON.parse(fs.readFileSync(metricsPath, 'utf-8'));
      res.status(200).json(metrics);
    } else {
      res.status(404).json({ detail: 'Metrics not found.' });
    }
  } catch (err) {
    res.status(500).json({ detail: 'Error loading metrics' });
  }
};
