# Genetic Disorder Prediction using SVM with Image Input

This invention presents a Support Vector Machine (SVM)-based model for the prediction of hereditary genetic disorders such as Hereditary Breast and Ovarian Cancer, Lynch Syndrome, and Familial Hypercholesterolemia. These are inherited disorders and usually need to be detected early to be treated and managed effectively. The model presented here uses Synthetic Minority Over-sampling Technique (SMOTE) to handle class imbalance, which is a prevalent problem in genetic datasets where rare disorders have much fewer samples. Through a balanced dataset, the model improves the accuracy and fairness of predictions. 

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

- data/: 
   Contains training images
- models/: 
   Stores trained model components
- dataPpreparation.py: 
   Image loading and preprocessing
- model_training.py: 
   Model training pipeline
- prediction_gui.py: 
   GUI for making predictions
- config.py: 
   Configuration constants
