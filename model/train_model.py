
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from catboost import CatBoostClassifier

# 1. Load dataset
df = pd.read_csv("dataset/AI4I_2020_Balanced_60_40.csv")

# 2. Select features and target
features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features].copy()
y = df["Machine failure"]

# 3. Categorical feature
categorical_features = ["Type"]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Create CatBoost model
model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    random_seed=42,
    loss_function="Logloss",
    verbose=False
)

# 6. Train model
model.fit(X_train, y_train, cat_features=categorical_features)

# 7. Evaluate model
y_pred = model.predict(X_test).astype(int).ravel()

print("\n========== CATBOOST EVALUATION ==========")
print("Accuracy :", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0) * 100, 2), "%")
print("Recall   :", round(recall_score(y_test, y_pred, zero_division=0) * 100, 2), "%")
print("F1 Score :", round(f1_score(y_test, y_pred, zero_division=0) * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 8. Save model
joblib.dump(model, "model/trained_model.pkl")

print("\nCatBoost model saved successfully!")
