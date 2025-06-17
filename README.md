
# 🛳️ Titanic Survival Prediction – Machine Learning Pipeline

![Titanic EDA Thumbnail](A_collection_of_four_data_visualizations_related_t.png)

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle](https://img.shields.io/badge/Kaggle-Titanic-blue)](https://www.kaggle.com/competitions/titanic)

This project tackles the Titanic survival prediction problem using advanced feature engineering, exploratory data analysis (EDA), and model ensembling to achieve high accuracy on the Kaggle leaderboard.

---

## 📁 Project Structure
```
titanic-project/
├── data/               # Raw dataset files (train.csv, test.csv)
├── eda_titanic.ipynb   # Notebook with detailed EDA
├── preprocess.py       # Data cleaning & feature engineering
├── train_model.py      # Model training & evaluation
├── make_submission.py  # Submission file generator
├── A_collection_of_four_data_visualizations_related_t.png  # Project thumbnail
└── README.md           # Project overview
```

---

## 🔍 Exploratory Data Analysis (EDA)
- Target variable (`Survived`) distribution
- Categorical feature interactions (Pclass, Sex, Embarked)
- Age and Fare distributions
- Correlation matrix and violin plots
- All plots are available in `eda_titanic.ipynb`

---

## 🛠️ Feature Engineering Highlights
- Title extraction from names
- Family size and group ID features
- Fare and Age binning
- Interaction terms like `Age × Pclass`, `FarePerPerson`, etc.
- Cabin deck and ticket prefix extraction

---

## 🤖 Models Trained
- Logistic Regression, Random Forest
- XGBoost and LightGBM with hyperparameter tuning
- Stacking and Voting Classifiers for ensemble modeling

**Best score:** `0.8329` on Kaggle Public Leaderboard

---

## 📦 Requirements
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run
```bash
python preprocess.py
python train_model.py
python make_submission.py
```

---

## 🏁 Final Notes
This project showcases a full ML workflow from raw data to leaderboard-ready model. It’s also a strong demonstration of **data storytelling, pipeline design, and reproducibility**.

---

## 📧 Contact
Created by [Alp Yaman](https://github.com/yourgithubusername) – feel free to reach out!

## 📄 License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
