# 🚀 Customer Churn Prediction API (FastAPI + XGBoost)

## 📌 Overview
This project is an **end-to-end Machine Learning API** for predicting **customer churn** in the banking sector.  
It is designed as a **production-style ML system**, not just a notebook model.

The API predicts whether a bank customer is likely to **leave (churn)** based on demographic, financial, and behavioral features.

The system follows **industry best practices**:
- Offline model training
- Serialized model artifact
- FastAPI-based inference service
- Publicly accessible REST API
- Clear evaluation and explainability

---

## 🏦 Dataset
- The dataset is based on **real-world banking data from a U.S. bank**
- Features include:
  - Customer demographics
  - Customer age
  - Account balance
  - Product usage
  - Credit score
  - Activity status

> ⚠️ Note:  
> The data has been cleaned and prepared for educational and portfolio demonstration purposes.

---

## 🧠 Machine Learning Model
- **Algorithm**: XGBoost Classifier
- **Why XGBoost?**
  - Handles non-linear relationships well
  - Strong performance on tabular data
  - Widely used in production ML systems

### Feature Engineering Includes:
- Log and square-root transformations
- Interaction features (e.g., balance × age)
- Behavioral flags (e.g., inactive older customers)

The trained model is saved as a serialized file (`.pkl`) and loaded at runtime for fast inference.

---

## 📊 Model Evaluation
Model evaluation is performed **offline** and documented in provided notebook.


### Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Stratified Cross-Validation
- Permutation Feature Importance

> ✅ Keeping evaluation separate from the API is an **industry-standard practice** to ensure:
> - Fast API response times
> - Stable deployments
> - Reproducible model analysis

---

## 🌐 API Deployment
The model is deployed as a **FastAPI application** and exposed through a **public URL**.

### 🔗 Base URL
https://customer-churn-api.up.railway.app

### 📘 Interactive API Docs (Swagger)
https://customer-churn-api.up.railway.app/docs


Clients can:
- View all endpoints
- Send test requests
- See input/output schemas
- Try predictions directly in the browser

---

## 🔮 Prediction Endpoint

### **POST** `/predict`

#### Request Body (JSON example):
```json
{
  "RowNumber": 1,
  "CustomerId": 15634602,
  "Surname": "Hargrave",
  "CreditScore": 619,
  "Geography": "France",
  "Gender": "Female",
  "Age": 42,
  "Tenure": 2,
  "Balance": 0.0,
  "NumOfProducts": 1,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 101348.88
}

#### Response Example
{
  "churn_probability": 0.73,
  "churn_prediction": 1
}

### Project Structure

customer-churn-api/
│
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── schemas.py           # Request/response validation
│   ├── model_loader.py      # Model loading logic
│
├── ml/
│   ├── churn_model.py       # ML model class
│   ├── train.py             # Training script
│
├── model/
│   └── churn_model.pkl      # Trained model artifact
│
├── notebooks/
│   └── customer_churn_evaluation.ipynb
│
├── requirements.txt
├── Procfile
└── README.md

## ⚙️ How the System Works (End-to-End)

1. The machine learning model is trained **offline** using historical customer data.
2. Model performance is **evaluated and validated** in a Jupyter Notebook using multiple metrics.
3. The trained model is **serialized and saved** as a reusable artifact (`.pkl` file).
4. The FastAPI application **loads the trained model at startup** for inference.
5. Clients send customer information to the API via a **POST request**.
6. The API processes the input data and **returns churn probability and prediction** in real time.

## ⚙️ How the System Works (End-to-End)

1. The machine learning model is trained **offline** using historical customer data.
2. Model performance is **evaluated and validated** in a Jupyter Notebook using multiple metrics.
3. The trained model is **serialized and saved** as a reusable artifact (`.pkl` file).
4. The FastAPI application **loads the trained model at startup** for inference.
5. Clients send customer information to the API via a **POST request**.
6. The API processes the input data and **returns churn probability and prediction** in real time.


