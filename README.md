# AI Predictive Maintenance System (RUL Prediction)
# 🚀 AI Predictive Maintenance System

LSTM 기반 항공기 엔진 RUL 예측 및 실시간 대시보드 시스템

## 📌 Overview
This project predicts the Remaining Useful Life (RUL) of aircraft engines using time-series sensor data.

It is based on the NASA C-MAPSS dataset and implements both traditional machine learning models and deep learning (LSTM).

A Streamlit dashboard is developed to visualize sensor trends and predict failure risk.

---

## 📊 Dataset
- NASA C-MAPSS FD001
- 100 engines
- 3 operational settings
- 21 sensors
- Total features: 24

---

## ⚙️ Problem
Predict Remaining Useful Life (RUL) based on sensor time-series data.

---

## 🤖 Models
- RandomForest Regressor
- XGBoost Regressor
- LSTM (Final Model)

---

## 📈 Performance (RMSE)

| Model | RMSE |
|------|------|
| RandomForest | 18.76 |
| XGBoost | 18.00 |
| LSTM | **5.69** |

LSTM significantly outperforms traditional ML models by capturing temporal dependencies.

---

## 🧠 Method
- Sliding Window (sequence length = 30)
- Time-series modeling using LSTM
- Sensor normalization and preprocessing

---

## 🖥️ Dashboard Features
- Engine selection
- Sensor trend visualization
- RUL prediction
- Failure risk estimation
- Prediction window visualization

---

## 🚀 Run

```bash
pip install -r requirements.txt
streamlit run app.py
