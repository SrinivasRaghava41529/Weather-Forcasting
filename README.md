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

<img width="939" height="378" alt="image" src="https://github.com/user-attachments/assets/175381dd-769a-49ca-b92b-7f8a61712c61" />

<img width="463" height="176" alt="image" src="https://github.com/user-attachments/assets/c63c35a9-37d1-40ea-87b7-36f4be224062" />

<img width="402" height="340" alt="image" src="https://github.com/user-attachments/assets/b84a9f62-8d1a-48f6-adee-e537ceab7864" />


---
