"""
Task 2: End-to-End ML Pipeline with Scikit-learn Pipeline API
Objective: Build a reusable and production-ready ML pipeline for predicting customer churn.
Dataset: Telco Customer Churn Dataset
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score)

# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────
def load_data(path: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv") -> pd.DataFrame:
    """Load Telco Churn dataset."""
    try:
        df = pd.read_csv(path)
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print("⚠️  Dataset not found locally. Generating synthetic data for demo...")
        df = generate_synthetic_data()
    return df


def generate_synthetic_data(n: int = 1000) -> pd.DataFrame:
    """Generate synthetic churn data for demo purposes."""
    np.random.seed(42)
    df = pd.DataFrame({
        "gender":           np.random.choice(["Male", "Female"], n),
        "SeniorCitizen":    np.random.choice([0, 1], n),
        "Partner":          np.random.choice(["Yes", "No"], n),
        "Dependents":       np.random.choice(["Yes", "No"], n),
        "tenure":           np.random.randint(0, 72, n),
        "PhoneService":     np.random.choice(["Yes", "No"], n),
        "MultipleLines":    np.random.choice(["Yes", "No", "No phone service"], n),
        "InternetService":  np.random.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity":   np.random.choice(["Yes", "No", "No internet service"], n),
        "TechSupport":      np.random.choice(["Yes", "No", "No internet service"], n),
        "Contract":         np.random.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": np.random.choice(["Yes", "No"], n),
        "PaymentMethod":    np.random.choice(["Electronic check", "Mailed check",
                                              "Bank transfer", "Credit card"], n),
        "MonthlyCharges":   np.round(np.random.uniform(20, 120, n), 2),
        "TotalCharges":     np.round(np.random.uniform(20, 8000, n), 2),
        "Churn":            np.random.choice(["Yes", "No"], n, p=[0.27, 0.73]),
    })
    return df


# ──────────────────────────────────────────────
# 2. PREPROCESS
# ──────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    """Clean and split features / target."""
    df = df.copy()

    # Drop customerID if present
    df.drop(columns=["customerID"], errors="ignore", inplace=True)

    # Fix TotalCharges (may be string with spaces)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Identify column types
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    print(f"   Numeric features  : {num_cols}")
    print(f"   Categorical features: {cat_cols}")

    return X, y, num_cols, cat_cols


# ──────────────────────────────────────────────
# 3. BUILD PIPELINES
# ──────────────────────────────────────────────
def build_pipeline(model, num_cols, cat_cols) -> Pipeline:
    """Build a full preprocessing + model pipeline."""

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])

    return pipeline


# ──────────────────────────────────────────────
# 4. HYPERPARAMETER TUNING
# ──────────────────────────────────────────────
def tune_pipeline(pipeline: Pipeline, X_train, y_train,
                  param_grid: dict, cv: int = 5) -> GridSearchCV:
    """Run GridSearchCV on the pipeline."""
    print("\n🔍 Running GridSearchCV ...")
    gs = GridSearchCV(pipeline, param_grid, cv=cv,
                      scoring="roc_auc", n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    print(f"   Best params : {gs.best_params_}")
    print(f"   Best CV AUC : {gs.best_score_:.4f}")
    return gs


# ──────────────────────────────────────────────
# 5. EVALUATE
# ──────────────────────────────────────────────
def evaluate(model, X_test, y_test, name: str = "Model"):
    """Print evaluation metrics."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*50}")
    print(f"📊 {name} Evaluation")
    print(f"{'='*50}")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# ──────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Task 2: End-to-End ML Pipeline — Customer Churn Prediction")
    print("=" * 60)

    # Load & split
    df = load_data()
    X, y, num_cols, cat_cols = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n📦 Train: {X_train.shape} | Test: {X_test.shape}")

    # ── Logistic Regression ──
    lr_pipeline = build_pipeline(LogisticRegression(max_iter=1000), num_cols, cat_cols)
    lr_param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__solver": ["lbfgs", "liblinear"],
    }
    print("\n\n🚀 Training Logistic Regression Pipeline ...")
    lr_gs = tune_pipeline(lr_pipeline, X_train, y_train, lr_param_grid)
    evaluate(lr_gs.best_estimator_, X_test, y_test, "Logistic Regression")

    # ── Random Forest ──
    rf_pipeline = build_pipeline(RandomForestClassifier(random_state=42), num_cols, cat_cols)
    rf_param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5],
    }
    print("\n\n🌲 Training Random Forest Pipeline ...")
    rf_gs = tune_pipeline(rf_pipeline, X_train, y_train, rf_param_grid)
    evaluate(rf_gs.best_estimator_, X_test, y_test, "Random Forest")

    # ── Cross-validation on best RF ──
    print("\n\n🔁 Cross-Validation (Random Forest best model):")
    cv_scores = cross_val_score(rf_gs.best_estimator_, X, y, cv=5, scoring="roc_auc")
    print(f"   CV AUC scores : {cv_scores}")
    print(f"   Mean AUC      : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Export pipelines ──
    joblib.dump(lr_gs.best_estimator_, "logistic_regression_pipeline.pkl")
    joblib.dump(rf_gs.best_estimator_, "random_forest_pipeline.pkl")
    print("\n✅ Pipelines exported:")
    print("   → logistic_regression_pipeline.pkl")
    print("   → random_forest_pipeline.pkl")

    # ── Load & quick predict demo ──
    loaded_rf = joblib.load("random_forest_pipeline.pkl")
    sample_pred = loaded_rf.predict(X_test.head(3))
    print(f"\n🎯 Sample predictions (first 3 test rows): {sample_pred.tolist()}")
    print("\n✅ Task 2 Complete!")


if __name__ == "__main__":
    main()
