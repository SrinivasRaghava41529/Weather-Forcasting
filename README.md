# 🌦️ ML-Based Weather Forecasting System

**Time-series machine learning system** for temperature forecasting using advanced feature engineering, multi-horizon prediction, and uncertainty estimation.

---

## 🚀 Key Features

* ⏱️ **Multi-horizon forecasting** (1h, 6h, 12h, 24h, 48h)
* 📊 **30+ engineered features** including:

  * Lag features (up to 48 hours)
  * Rolling statistics (mean, std)
  * Cyclical time encoding (sin/cos)
* 🌡️ **Derived meteorological features**
* 📉 **Uncertainty estimation** using model error (±MAE)
* 🧠 **Model interpretability ready** (LIME/SHAP integration)
* ⚡ Interactive UI (Streamlit)

---

## 🧠 Problem Statement

Weather prediction is inherently temporal and depends on historical patterns.
This project builds a machine learning pipeline that captures:

* Short-term dependencies (lag features)
* Trends and variability (rolling statistics)
* Seasonal patterns (cyclical encoding)

---

## 🏗️ System Architecture

    Raw Inputs (UI)
    ↓
    Feature Engineering Pipeline
    ↓
    ML Model (Trained on 30+ features)
    ↓
    Prediction + Confidence Interval

---

## 🔧 Feature Engineering

The model does not rely only on current inputs. It automatically derives:

### 1. Lag Features

* `temp_c_lag_1h`, `temp_c_lag_24h`, etc.

### 2. Rolling Statistics

* Moving averages and standard deviations over multiple windows

### 3. Time Encoding

* `hour_sin`, `hour_cos`
* `month_sin`, `month_cos`

### 4. Derived Features

* Vapor pressure
* Wind vector components
* Pressure trends

---

## 📊 Model Performance

* Cross-validation MAE: **~0.38°C**
* Robust across multiple forecast horizons

---

## 📈 Example Output

* Predicted Temperature: **13.2°C**
* Confidence Range: **±1.5°C**
* Trend visualization over 48 hours

---

## 🖥️ UI Preview

(Add your screenshot here)

---

## ⚙️ Tech Stack

* Python
* Scikit-learn / XGBoost / LightGBM (your model)
* Pandas, NumPy
* Streamlit
* SHAP / LIME (optional)

---

## ▶️ How to Run

```bash
git clone <repo-url>
cd project
pip install -r requirements.txt
streamlit run app.py
```

---

## 📌 Key Learnings

* Importance of **feature engineering in time-series ML**
* Handling **training vs inference consistency**
* Designing **ML pipelines for real-world deployment**
* Communicating model uncertainty

---

## 🔮 Future Improvements

* Add SHAP-based explainability dashboard
* Integrate real-time weather API
* Deploy as REST API (FastAPI)

---

## 👤 Author

Your Name
