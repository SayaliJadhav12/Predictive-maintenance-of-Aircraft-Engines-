# Predictive Maintenance of Aircraft Engines (PdMS)

## 📌 Overview
This project is a Proof of Concept (PoC) for a **Predictive Maintenance System (PdMS)** using multivariate time-series sensor data.
It explores and compares **1D CNN**, **LLM (T5/Chronos-style)**, and a **Hybrid CNN + LLM approach** to predict engine health, detect failures, and perform short-term forecasting.

The project also includes **interactive Streamlit dashboards** for training, inference, and forecasting.

---

## 🎯 Objectives
- Predict engine health (Healthy / Failure)
- Compare CNN-only, LLM-only, and Hybrid models
- Extract CNN embeddings for long-range reasoning
- Perform unit-wise (engine-level) evaluation
- Build training, inference, and forecasting dashboards

---

## 🏗️ Architecture
**High-level flow:**

Sensor Data → Preprocessing → 1D CNN → Embeddings → T5 LLM → Prediction & Forecast

---

## 🚀 Features Implemented
- Time-series windowing (15-cycle context)
- 1D CNN for sensor pattern learning
- CNN embedding extraction
- LLM-based binary classification (T5-small)
- Hybrid CNN + LLM model
- Unit-wise evaluation
- Streamlit dashboards:
  - Data preparation & training
  - Inference & forecasting
- Model performance visualization (ROC, PR, Confusion Matrix, PCA)

---

## 🧩 Challenges & Learnings
Handling long-term dependencies in sensor data
Combining CNN embeddings with LLM input sequences
Visualizing sensor-level contributions for explainability
Maintaining small dashboard latency with multiple models

---

## 🔮 Future Work
Integrate real-time sensor streaming
Add predictive RUL (Remaining Useful Life) forecasting
Deploy dashboard on cloud for multi-user access
Fine-tune LLM on larger datasets for higher accuracy

---

## 📈 Results
Metrics: Accuracy, F1-score, Precision, Recall
Visualizations: ROC, PR curve, confusion matrix, PCA
Observations: Hybrid model generally outperforms standalone CNN or LLM
