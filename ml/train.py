import os
import pandas as pd
import joblib
from ml.churn_model import ChurnModel

# Ensure model directory exists

# Load dataset
data = pd.read_csv("data/Churn_Modelling.csv")

X = data.drop(columns=["Exited"])
y = data["Exited"]

model = ChurnModel()
model.fit(X, y)

# Save trained model
joblib.dump(model, "model/churn_model.pkl")

print("Model trained and saved successfully.")
