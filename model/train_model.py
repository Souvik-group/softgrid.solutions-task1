import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "student_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "student_model.pkl")

df = pd.read_csv(DATASET_PATH)
print("Grade distribution:\n", df["Grade"].value_counts())

X = df[["StudyHours", "Attendance", "Assignments", "PreviousMarks", "Participation"]]
y = df["Grade"]

le = LabelEncoder()
y_enc = le.fit_transform(y)
print("Classes:", le.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"\nAccuracy: {acc * 100:.2f}%")
print(classification_report(y_test, pred, target_names=le.classes_))

# Regression-style metrics (treat encoded labels as numeric)
y_test_num = y_test
y_pred_num = pred
mae = mean_absolute_error(y_test_num, y_pred_num)
mse = mean_squared_error(y_test_num, y_pred_num)
r2 = r2_score(y_test_num, y_pred_num)
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

with open(MODEL_PATH, "wb") as f:
    pickle.dump(
        {
            "model": model,
            "label_encoder": le,
            "feature_columns": list(X.columns),
            "metrics": {
                "accuracy": float(acc),
                "mae": float(mae),
                "mse": float(mse),
                "r2": float(r2),
            },
        },
        f,
    )
print(f"Model saved to {MODEL_PATH}")
