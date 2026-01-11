import joblib
import pandas as pd

model = joblib.load("model/churn_model.pkl")

sample = {
    "RowNumber": 1,
    "CustomerId": 15634602,
    "Surname": "Hargrave",
    "Geography": "France",
    "Gender": "Female",
    "Age": 42,
    "CreditScore": 619,
    "Balance": 0.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 101348.88
}

df = pd.DataFrame([sample])

model.predict_proba(df)
