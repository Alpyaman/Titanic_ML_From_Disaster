"""
Titanic Kaggle Submission Generator

This script generates predictions for the Kaggle Titanic competition test set
using a trained XGBoost model with optimized hyperparameters.

Usage:
    python make_submission.py

Output:
    Creates 'submission.csv' file ready for Kaggle upload.

Author: Alp Yaman
Date: 2025
"""

import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from preprocess import preprocess  # your preprocessing function

# ========== 1. Load and preprocess training data ==========

# Load training data and apply preprocessing pipeline
train_df = preprocess('titanic_data/train.csv')
train_df.fillna(train_df.median(numeric_only=True), inplace=True)

# Separate features and target variable
X_train = train_df.drop(columns=['Survived', 'PassengerId'])
y_train = train_df['Survived']

# ========== 2. Load and preprocess test data ==========

# Apply same preprocessing pipeline to test data
test_df = preprocess('titanic_data/test.csv')
test_df.fillna(test_df.median(numeric_only=True), inplace=True)

# Save PassengerId for final submission file
passenger_ids = test_df['PassengerId']

# Remove PassengerId from test features before prediction
X_test = test_df.drop(columns=['PassengerId'])

# ========== 3. Feature scaling ==========

# Standardize features using same scaler fitted on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== 4. Train XGBoost model with optimized parameters ==========

# Initialize XGBoost with best hyperparameters from GridSearch
xgb_best = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=100,  # Number of boosting rounds
    max_depth=5,       # Maximum tree depth
    learning_rate=0.1, # Step size shrinkage
    random_state=42    # Reproducibility
)

# Train model on scaled training data
xgb_best.fit(X_train_scaled, y_train)

# ========== 5. Generate predictions ==========

# Predict survival for test set
predictions = xgb_best.predict(X_test_scaled)

# ========== 6. Create submission file ==========

# Format predictions according to Kaggle requirements
submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': predictions
})

# Save to CSV (no index column)
submission.to_csv('submission.csv', index=False)

print("✅ Submission file saved as submission.csv")
print(f"📊 Total predictions: {len(submission)}")
print(f"🎯 Predicted survivors: {predictions.sum()} ({predictions.sum()/len(predictions)*100:.1f}%)")
