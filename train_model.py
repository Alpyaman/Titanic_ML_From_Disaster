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
    joblib.dump(model, filepath)
    print(f"💾 Model saved to {filepath}")

def save_metadata(results, filepath="model_metadata.json"):
    meta = {}
    for name, score, model in results:
        meta[name] = {"cv_accuracy": score}
    with open(filepath, 'w') as f:
        json.dump(meta, f, indent=4)
    print(f"📁 Metadata saved to {filepath}")

if __name__ == "__main__":
    print("🚀 Starting training pipeline...")
    df = preprocess('C:/Users/alpya/Documents/titanic-project/titanic_data/train.csv')

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

    save_model(stacking_model, "scripts/best_stacking_model.pkl")
    save_metadata(results)

    # Final report on training data
    y_pred = stacking_model.predict(X)
    print("\n📋 Classification Report on Training Data:")
    print(classification_report(y, y_pred))

    print("🎉 Training pipeline complete!")
    print("✅ All models trained and saved successfully.")