#!/usr/bin/env bash
# exit on error
set -o errexit

# 1. Install all the Python packages
echo "📦 Installing requirements..."
pip install -r requirements.txt

# 2. Run the training script to generate the .pkl model files
echo "🧠 Training the machine learning model..."
python train_model.py

echo "✅ Build complete."