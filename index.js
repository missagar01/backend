const express = require('express');
const cors = require('cors');
require('dotenv').config();
const connectDB = require('./config/database');

const authRoutes = require('./routes/authRoutes');
const predictRoutes = require('./routes/predictRoutes');

const app = express();

app.use(cors());
app.use(express.json());

// Routes
app.use('/api/auth', authRoutes);
app.use('/api', predictRoutes);

app.get('/', (req, res) => {
  res.json({ message: "Welcome to Context-Aware Multi-Disease Detection System API (Node.js + MongoDB)" });
});

const PORT = process.env.PORT || 8000;

// Start server first, then connect to DB (non-blocking startup)
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}.`);
  // Connect to MongoDB Atlas after server starts
  connectDB();
});
