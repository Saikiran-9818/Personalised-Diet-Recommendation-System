#!/usr/bin/env bash
set -o errexit  

# packages
pip install -r requirements.txt

# first train model
python train_model.py

# run the app
python app.py
