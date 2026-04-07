const express = require('express');
const router = express.Router();
const predictController = require('../controllers/predictController');
const { authMiddleware } = require('../controllers/authController');

router.post('/predict', authMiddleware, predictController.predict);
router.get('/history', authMiddleware, predictController.getHistory);

module.exports = router;
