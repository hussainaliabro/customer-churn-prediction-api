import joblib

# Load model once at startup
model = joblib.load("model/churn_model.pkl")
