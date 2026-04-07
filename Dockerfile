# Use a professional image that has BOTH Python and Node.js
FROM nikolaik/python-nodejs:python3.11-nodejs18

# Set working directory
WORKDIR /app

# Copy package files and install Node.js dependencies
COPY package*.json ./
RUN npm install

# Copy the rest of the backend code (including ml_models)
COPY . .

# Install Python dependencies strictly
RUN pip install --no-cache-dir numpy==1.26.4 pandas==2.2.2 scikit-learn==1.4.2 xgboost==2.0.3 joblib==1.4.2

# Expose the port your app runs on
EXPOSE 8000

# Start the application
CMD ["node", "index.js"]
