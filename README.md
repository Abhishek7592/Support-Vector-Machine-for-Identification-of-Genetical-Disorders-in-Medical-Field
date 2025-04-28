# Genetic Disorder Prediction using SVM with Image Input

This project implements an SVM-based classifier to predict genetic disorders from image data.

## Features
- Image preprocessing and feature extraction
- Handling class imbalance with SMOTE
- Hyperparameter tuning with GridSearchCV
- GUI for making predictions

## Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

## File Structure

data/: 
   Contains training images
models/: 
   Stores trained model components
dataPpreparation.py: 
   Image loading and preprocessing
model_training.py: 
   Model training pipeline
prediction_gui.py: 
   GUI for making predictions
config.py: 
   Configuration constants
