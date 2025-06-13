import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from preprocess import preprocess  # your preprocessing function

# 1. Load and preprocess training data
train_df = preprocess('C:/Users/alpya/Documents/titanic-project/titanic_data/train.csv')
train_df.fillna(train_df.median(numeric_only=True), inplace=True)

X_train = train_df.drop(columns=['Survived', 'PassengerId'])
y_train = train_df['Survived']

# 2. Load and preprocess test data (make sure you preprocess similarly!)
test_df = preprocess('C:/Users/alpya/Documents/titanic-project/titanic_data/test.csv')
test_df.fillna(test_df.median(numeric_only=True), inplace=True)

# Save PassengerId for submission
passenger_ids = test_df['PassengerId']

# Remove PassengerId from test features before prediction
X_test = test_df.drop(columns=['PassengerId'])
X_test = test_df.drop(columns=['PassengerId'])

# 3. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train XGBoost model with your best parameters
xgb_best = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=100,  # set your best n_estimators here
    max_depth=5,       # your best max_depth
    learning_rate=0.1, # your best learning_rate
    random_state=42
)

xgb_best.fit(X_train_scaled, y_train)

# 5. Predict on test set
predictions = xgb_best.predict(X_test_scaled)

# 6. Prepare submission DataFrame
submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': predictions
})

# 7. Save submission CSV
submission.to_csv('submission.csv', index=False)

print("✅ Submission file saved as submission.csv")
