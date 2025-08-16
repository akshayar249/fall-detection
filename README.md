

---

# 🚨 Fall Detection System

## 📌 Overview

This project implements a **Fall Detection System** using **accelerometer and gyroscope sensor data** to detect falls in real time.

The system leverages a **stacked machine learning model** (KNN + Logistic Regression as base models, Random Forest as the meta model) to classify activities into:

* 🟥 **Fall**
* 🟩 **Not Fall**

When a fall is detected, the system can:

* Send **alert notifications** to emergency contacts 📲
* Trigger an **audio buzzer/alarm** 🔊 for immediate assistance

---

## ⚡ Key Features

* 📱 **Sensor Data Acquisition**: Uses smartphone accelerometer & gyroscope
* 🧮 **Feature Extraction**: Mean, standard deviation, skewness, kurtosis, max, min
* 🤖 **Stacked Machine Learning Model**:

  * Base Models → **KNN**, **Logistic Regression**
  * Meta Model → **Random Forest**
* 🚑 **Emergency Response**: Notifications & buzzer alerts

---

## 📊 Results

| Model                             | Accuracy | Precision | Recall  | Key Strengths                         |
| --------------------------------- | -------- | --------- | ------- | ------------------------------------- |
| **KNN (k=5)**                     | 93%      | 92%       | 91%     | Simple, effective, local patterns     |
| **Logistic Regression**           | 94%      | 94%       | 92%     | Fast, interpretable                   |
| **Random Forest**                 | 95%      | 94%       | 93%     | Robust, ensemble learning             |
| **Stacked Model (KNN + LR → RF)** | **96%**  | **95%**   | **94%** | Best balance of accuracy & robustness |

---

## 🔮 Future Scope

* ☁️ Deploy backend on **cloud platforms** for remote monitoring
* 📍 Integrate with **GPS** to send location during a fall event
* ⌚ Extend support for **wearables** (e.g., ESP32, smartwatches)
* 🧠 Explore **deep learning (CNN/LSTM)** for improved classification
* 🖥️ Add a **caregiver dashboard** for real-time monitoring


