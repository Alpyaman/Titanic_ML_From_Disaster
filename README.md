# Titanic_ML_From_Disaster

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
