# Titanic Machine Learning From Disaster

This repository contains a machine learning pipeline to predict passenger survival on the Titanic dataset using various classifiers and ensemble methods. The project includes data preprocessing, model training with hyperparameter tuning, and generating Kaggle submission files.

---

## Project Overview

The goal is to build a predictive model that classifies whether a passenger survived the Titanic disaster based on features like age, fare, passenger class, and family size.

---

## Features

- Comprehensive preprocessing pipeline (`preprocess.py`)
- Multiple models including Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, Gradient Boosting
- Hyperparameter tuning using `GridSearchCV`
- Ensemble methods such as Stacking and Voting classifiers
- Model persistence using `joblib`
- Ready-to-submit Kaggle prediction pipeline

---

## Getting Started

### Prerequisites

- Python 3.8+
- Packages listed in `requirements.txt` (create with `pip freeze > requirements.txt`)

### Install Dependencies

```bash
pip install -r requirements.txt
```
This file includes the main libraries used in this project such as:
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- joblib

## Usage
### Preprocess Data
```bash
from preprocess import preprocess

df_train = preprocess('data/train.csv')
df_test = preprocess('data/test.csv')
```
### Train Models with Hyperparameter Tuning
Run the training script:
```bash
python train_model.py
```
This will:
- Preprocess data
- Train multiple models using GridSearchCV
- Output best model and CV scores
- Save the best model as `best_model.pkl`

### Generate Kaggle Submission
Run the submission script:
```bash
python final_kaggle_submission.py
```
This script will:
- Load the test data
- Apply trained model
- Generate `submission.csv` ready for Kaggle upload

### Results
- Best cross-validation accuracy achieved: ~0.84 (XGBoost)
- Final Kaggle submission score: 0.75119

## Project Structure
```bash
.
├── preprocess.py                  # Data preprocessing functions
├── train_model.py                 # Model training and tuning pipeline
├── final_kaggle_submission.py     # Script for generating Kaggle submission CSV
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── titanic_data/
    ├── train.csv                  # Titanic training dataset
    └── test.csv                   # Titanic test dataset
```
## License
This project is licensed under the MIT License.

## Acknowledgments
- Kaggle Titanic Competition
- Machine learning libraries: scikit-learn,XGBoost, LightGBM
