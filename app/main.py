from fastapi import FastAPI
import pandas as pd
from app.schemas import ChurnInput
from app.model_loader import model

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether a customer will churn based on account details",
    version="1.0"
)

@app.post("/predict")
def predict_churn(data: ChurnInput):
    """
    Receives customer data and returns churn prediction.
    """

    # Convert incoming JSON into DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Get churn probability
    probability = model.predict_proba(input_df)[0]

    # Apply business threshold
    prediction = int(probability >= 0.3)

    return {
        "churn_probability": round(float(probability), 3),
        "churn_prediction": prediction
    }
