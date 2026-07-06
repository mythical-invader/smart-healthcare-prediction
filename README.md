# 🩺 Smart Healthcare Prediction System

A modern **Machine Learning-powered Healthcare Web Application** that predicts whether a patient is **Non-Diabetic**, **Pre-Diabetic**, or **Diabetic** using physiological measurements and an Ensemble Learning model.

---

## ✨ Features

- **🤖 AI-Powered Prediction** — Soft Voting Ensemble combining Random Forest, Gradient Boosting and Logistic Regression.
- **⚡ Real-Time Analysis** — Instantly predicts diabetes risk through a Flask REST API.
- **🧠 Intelligent Feature Engineering** — Automatically replaces biological zero-values, creates interaction features and prepares data before prediction.
- **📊 Confidence Scores** — Returns probability for each class.
- **💬 Personalized Recommendation** — Displays health recommendations based on prediction.
- **📱 Responsive Interface** — Clean HTML/CSS/JavaScript frontend.

---

## 💻 Tech Stack

- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Backend:** Python, Flask, Flask-CORS
- **Machine Learning:** Scikit-Learn, Random Forest, Gradient Boosting, Logistic Regression
- **Data Processing:** Pandas, NumPy
- **Model Training:** Jupyter Notebook
- **Utilities:** Joblib, Imbalanced-Learn (SMOTE)

---

## 📂 Project Structure

```
Smart-Healthcare-Prediction/
│
├── index.html
├── app.py
├── diabetes-prediction.ipynb
├── diabetes.csv
├── diabetes_model.pkl
├── scaler.pkl
└── README.md
```

| File | Description |
|------|-------------|
| `index.html` | User Interface |
| `app.py` | Flask Backend API |
| `diabetes-prediction.ipynb` | Data Cleaning & Model Training |
| `diabetes.csv` | Dataset |
| `diabetes_model.pkl` | Trained Model |
| `scaler.pkl` | StandardScaler |

---

## 🚀 How to Run Locally

### 1. Clone Repository

```bash
git clone https://github.com/mythical-invader/smart-healthcare-prediction.git

cd smart-healthcare-prediction
```

### 2. Install Dependencies

```bash
pip install flask flask-cors pandas numpy scikit-learn imbalanced-learn joblib jupyter
```

### 3. Train the Model

Open

```text
diabetes-prediction.ipynb
```

Run every cell.

This automatically generates:

```
diabetes_model.pkl
scaler.pkl
```

---

### 4. Start Flask Server

```bash
python app.py
```

Backend runs on

```
http://127.0.0.1:5000
```

---

### 5. Launch Frontend

Simply open

```text
index.html
```

in your browser and click **Analyze Patient Risk**.

---

## 📊 Example Test Cases

| Feature | 🟢 Non-Diabetic | 🟡 Pre-Diabetic | 🔴 Diabetic |
|---------|:---------------:|:---------------:|:-----------:|
| Pregnancies | 1 | 2 | 6 |
| Glucose | 85 | 115 | 150 |
| Blood Pressure | 66 | 75 | 80 |
| Skin Thickness | 20 | 30 | 35 |
| Insulin | 80 | 130 | 140 |
| BMI | 26.0 | 31.0 | 35.0 |
| Diabetes Pedigree | 0.20 | 0.50 | 0.80 |
| Age | 25 | 40 | 55 |

---

## 🌐 API Reference

The frontend communicates with the Flask backend using:

```http
POST /predict
```

### Request

```json
{
  "features": [
    2,
    120,
    75,
    28,
    130,
    31.5,
    0.52,
    40
  ]
}
```

### Response

```json
{
  "prediction": "Pre-Diabetic",
  "confidence": {
    "Non-Diabetic": 0.18,
    "Pre-Diabetic": 0.73,
    "Diabetic": 0.09
  },
  "recommendation": "Lifestyle modifications and regular monitoring are recommended."
}
```

---

## ⚙️ Prediction Workflow

```
Patient Input
      │
      ▼
Frontend (HTML/CSS/JS)
      │
      ▼
Flask API
      │
      ▼
Feature Engineering
      │
      ▼
Feature Scaling
      │
      ▼
Soft Voting Ensemble Model
      │
      ▼
Prediction + Confidence
      │
      ▼
Medical Recommendation
```

---

## 🔬 Machine Learning Pipeline

- Clinical Dataset
- Data Cleaning
- Missing Value Imputation
- Feature Engineering
- SMOTE Balancing
- Standard Scaling
- Soft Voting Ensemble Training
- Model Serialization (`.pkl`)
- Flask Deployment

---

## 🚧 Future Improvements

- User Authentication
- Patient History Dashboard
- PDF Report Generation
- Cloud Deployment
- Explainable AI (SHAP/LIME)
- Multi-Disease Prediction

---

## ⭐ Support

If you found this project useful, consider giving it a **Star ⭐** on GitHub.
