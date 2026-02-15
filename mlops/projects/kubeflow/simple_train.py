from pathlib import Path
import json
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
report = classification_report(y_test, preds, output_dict=True)

print(f"Accuracy: {acc:.4f}")

# Write metrics artifact
artifact_dir = Path("/home/jovyan/artifacts")
artifact_dir.mkdir(parents=True, exist_ok=True)
artifact_path = artifact_dir / "metrics.json"
artifact_path.write_text(json.dumps({"accuracy": acc, "report": report}, indent=2))

print("Wrote metrics:", artifact_path)

# Save model artifact
model_path = artifact_dir / "model.joblib"
joblib.dump(model, model_path)
print("Saved model:", model_path)
