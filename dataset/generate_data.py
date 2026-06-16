import pandas as pd
import numpy as np
import os

np.random.seed(42)

rows = []

# Generate ~200 samples per grade by controlling input ranges
grade_configs = {
    "A": dict(study=(6, 8),  attendance=(85, 100), assignments=(8, 10), marks=(80, 95),  participation=(7, 10), n=200),
    "B": dict(study=(4, 7),  attendance=(70, 90),  assignments=(6, 9),  marks=(65, 85),  participation=(5, 8),  n=200),
    "C": dict(study=(3, 5),  attendance=(60, 75),  assignments=(4, 7),  marks=(50, 70),  participation=(3, 6),  n=200),
    "D": dict(study=(2, 4),  attendance=(50, 65),  assignments=(2, 5),  marks=(40, 55),  participation=(2, 4),  n=200),
    "F": dict(study=(1, 2),  attendance=(50, 58),  assignments=(1, 3),  marks=(35, 45),  participation=(1, 3),  n=200),
}

for grade, cfg in grade_configs.items():
    n = cfg["n"]
    study        = np.random.randint(cfg["study"][0],        cfg["study"][1] + 1,        n)
    attendance   = np.random.randint(cfg["attendance"][0],   cfg["attendance"][1] + 1,   n)
    assignments  = np.random.randint(cfg["assignments"][0],  cfg["assignments"][1] + 1,  n)
    marks        = np.random.randint(cfg["marks"][0],        cfg["marks"][1] + 1,        n)
    participation= np.random.randint(cfg["participation"][0],cfg["participation"][1] + 1,n)

    for i in range(n):
        rows.append([study[i], attendance[i], assignments[i], marks[i], participation[i], grade])

df = pd.DataFrame(rows, columns=["StudyHours", "Attendance", "Assignments", "PreviousMarks", "Participation", "Grade"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

out_path = os.path.join(os.path.dirname(__file__), "student_data.csv")
df.to_csv(out_path, index=False)
print(f"Dataset created: {len(df)} records")
print(df["Grade"].value_counts().sort_index())
