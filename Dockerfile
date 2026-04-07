# Use a professional image that has BOTH Python and Node.js
FROM nikolaik/python-nodejs:python3.11-nodejs18

# Set working directory to /app
WORKDIR /app

# Copy package files and install Node.js dependencies
COPY package*.json ./
RUN npm install

# Copy everything from current directory to /app
COPY . .

# Install Python dependencies strictly
RUN pip install --no-cache-dir numpy==1.26.4 pandas==2.2.2 scikit-learn==1.4.2 xgboost==2.0.3 joblib==1.4.2

# Explicitly set the environment to production
ENV NODE_ENV=production
ENV PORT=8000

# Expose the port
EXPOSE 8000

# CRITICAL START COMMAND: node index.js
CMD ["node", "index.js"]
