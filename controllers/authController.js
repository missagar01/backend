const User = require('../models/User');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');

const SECRET_KEY = process.env.SECRET_KEY || 'supersecretkey_mtech_project_top_tier_2026';

exports.register = async (req, res) => {
  try {
    const { username, email, password } = req.body;
    
    const existingEmail = await User.findOne({ email });
    if (existingEmail) {
      return res.status(400).json({ detail: 'Email already registered' });
    }
    
    const existingUsername = await User.findOne({ username });
    if (existingUsername) {
      return res.status(400).json({ detail: 'Username already taken' });
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    const user = await User.create({ username, email, password: hashedPassword });

    res.status(201).json({ id: user._id, username: user.username, email: user.email });
  } catch (error) {
    res.status(500).json({ detail: 'Server Error', error: error.message });
  }
};

exports.login = async (req, res) => {
  try {
    const { username, password } = req.body;
    const user = await User.findOne({ username });
    if (!user) {
      return res.status(401).json({ detail: 'Incorrect username or password' });
    }

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(401).json({ detail: 'Incorrect username or password' });
    }

    const token = jwt.sign({ id: user._id, username: user.username }, SECRET_KEY, { expiresIn: '30d' });
    res.status(200).json({ access_token: token, token_type: 'bearer' });
  } catch (error) {
    res.status(500).json({ detail: 'Server Error' });
  }
};

exports.authMiddleware = async (req, res, next) => {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ detail: 'No token provided' });
  }
  const token = authHeader.split(' ')[1];
  try {
    const decoded = jwt.verify(token, SECRET_KEY);
    req.user = await User.findById(decoded.id);
    if (!req.user) {
      return res.status(401).json({ detail: 'Invalid token' });
    }
    next();
  } catch (err) {
    res.status(401).json({ detail: 'Unauthorized' });
  }
};
