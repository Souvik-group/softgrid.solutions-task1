from flask import Flask, render_template, request
import os
import pickle
import pandas as pd

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "student_model.pkl")

app = Flask(__name__)

with open(MODEL_PATH, "rb") as f:
    payload = pickle.load(f)

model = payload["model"]
label_encoder = payload["label_encoder"]
feature_columns = payload["feature_columns"]


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    values = {}

    if request.method == "POST":
        values = {
            "study":         request.form.get("study", ""),
            "attendance":    request.form.get("attendance", ""),
            "assignments":   request.form.get("assignments", ""),
            "marks":         request.form.get("marks", ""),
            "participation": request.form.get("participation", ""),
        }

        input_df = pd.DataFrame([{
            "StudyHours":    float(values["study"]),
            "Attendance":    float(values["attendance"]),
            "Assignments":   float(values["assignments"]),
            "PreviousMarks": float(values["marks"]),
            "Participation": float(values["participation"]),
        }], columns=feature_columns)

        pred_enc = model.predict(input_df)[0]
        prediction = label_encoder.inverse_transform([pred_enc])[0]

    return render_template("index.html", prediction=prediction, values=values)


if __name__ == "__main__":
    app.run(debug=True)
