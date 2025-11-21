"""
Titanic Model Training Module

This module handles model training, hyperparameter tuning, and ensemble creation
for the Titanic survival prediction problem.

Features:
- Multiple ML algorithms with GridSearchCV
- Stacking ensemble with top-performing models
- Cross-validation with StratifiedKFold
- Model persistence and metadata tracking

Author: Alp Yaman
Date: 2025
"""

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from preprocess import preprocess
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

def get_models_and_params():
    """
    Define machine learning models and hyperparameter grids for tuning.

    Returns
    -------
    dict
        Dictionary mapping model names to their configurations.
        Each configuration contains:
        - 'model': instantiated model object
        - 'params': hyperparameter grid for GridSearchCV

    Examples
    --------
    >>> models = get_models_and_params()
    >>> print(models.keys())
    dict_keys(['Logistic Regression', 'Decision Tree', ...])
    """
    
    return {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                "C": [0.01, 0.1, 1, 10],
                "solver": ["liblinear", "lbfgs"]
            }
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10]
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5]
            }
        },
        "XGBoost": {
            "model": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 10],
                "learning_rate": [0.01, 0.1, 0.2]
            }
        },
        "LightGBM": {
            "model": LGBMClassifier(verbose=-1, random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [-1, 5, 10],
                "learning_rate": [0.01, 0.1, 0.2]
            }
        },
        "CatBoost": {
            "model": CatBoostClassifier(verbose=0, random_state=42),
            "params": {
                "iterations": [100, 200],
                "depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2]
            }
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 10]
            }
        }
    }

def train_with_gridsearch(X, y):
    """
    Train multiple models with hyperparameter tuning using GridSearchCV.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix for training.
    y : pd.Series or np.ndarray
        Target variable (survival labels).

    Returns
    -------
    tuple
        (best_model, results) where:
        - best_model: fitted model with highest CV accuracy
        - results: list of (name, score, estimator) tuples for all models

    Notes
    -----
    - Uses StandardScaler for feature normalization
    - Performs 5-fold cross-validation
    - Searches hyperparameter grid for each model
    - Prints progress and results during training
    """
    
    models = get_models_and_params()
    results = []
    best_score = 0
    best_model = None
    best_name = ""

    for name, cfg in models.items():
        print(f"\n🔍 Training {name} with GridSearchCV...")
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', cfg["model"])
        ])
        param_grid = {f'clf__{k}': v for k, v in cfg['params'].items()}

        grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X, y)

        print(f"✅ Best Params for {name}: {grid.best_params_}")
        print(f"📈 CV Accuracy: {grid.best_score_:.4f}")

        results.append((name, grid.best_score_, grid.best_estimator_))

        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_
            best_name = name

    print(f"\n🏆 Best Model: {best_name} with mean CV accuracy {best_score:.4f}")

    print("\n📊 Results Summary:")
    for model_name, score, estimator in results:
        print(f"{model_name}: CV Accuracy = {score:.4f}, Estimator = {estimator}")
    return best_model, results

def build_stacking_ensemble(best_estimators, X, y):
    """
    Build and train a stacking ensemble from best-performing models.

    Parameters
    ----------
    best_estimators : list of tuples
        List of (name, estimator) tuples for base models.
    X : pd.DataFrame or np.ndarray
        Feature matrix for training.
    y : pd.Series or np.ndarray
        Target variable (survival labels).

    Returns
    -------
    StackingClassifier
        Fitted stacking ensemble with GradientBoosting meta-learner.

    Notes
    -----
    - Uses GradientBoosting as meta-learner for final predictions
    - Performs 5-fold stratified cross-validation
    - Includes passthrough=True to give meta-learner access to base features
    """
    
    # Use GradientBoosting as meta-learner
    stacking = StackingClassifier(
        estimators=best_estimators,
        final_estimator=GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=42),
        cv=5,
        n_jobs=-1,
        passthrough=True
    )
    print("\n🔄 Training Stacking Ensemble with Stratified 5-Fold CV...")
    cv_scores = cross_val_score(stacking, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='accuracy', n_jobs=-1)
    print(f"Stacking Ensemble CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    stacking.fit(X, y)
    return stacking

def save_model(model, filepath):
    """
    Save trained model to disk using joblib.

    Parameters
    ----------
    model : estimator object
        Trained scikit-learn compatible model.
    filepath : str
        Path where model will be saved.
    """
    joblib.dump(model, filepath)
    print(f"💾 Model saved to {filepath}")

def save_metadata(results, filepath="model_metadata.json"):
    """
    Save model performance metrics to JSON file.

    Parameters
    ----------
    results : list of tuples
        List of (name, score, model) tuples from training.
    filepath : str, default="model_metadata.json"
        Path where metadata will be saved.
    """
    meta = {}
    for name, score, model in results:
        meta[name] = {"cv_accuracy": score}
    with open(filepath, 'w') as f:
        json.dump(meta, f, indent=4)
    print(f"📁 Metadata saved to {filepath}")

if __name__ == "__main__":
    print("🚀 Starting training pipeline...")
    df = preprocess('titanic_data/train.csv')

    print("\n🔎 Missing values after preprocessing:")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    df.fillna(df.median(numeric_only=True), inplace=True)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    best_model, results = train_with_gridsearch(X, y)

    # Select top 4 estimators for stacking
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    top_estimators = [(name, model.named_steps['clf']) for name, _, model in sorted_results[:4]]

    stacking_model = build_stacking_ensemble(top_estimators, X, y)

    save_model(stacking_model, "best_stacking_model.pkl")
    save_metadata(results)

    # Final report on training data
    y_pred = stacking_model.predict(X)
    print("\n📋 Classification Report on Training Data:")
    print(classification_report(y, y_pred))

    print("🎉 Training pipeline complete!")

    print("✅ All models trained and saved successfully.")

