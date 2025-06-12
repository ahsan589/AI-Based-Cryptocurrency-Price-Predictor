# 🧠 AI-Based Cryptocurrency Price Predictor API

This is a Flask-based REST API that uses an **LSTM (Long Short-Term Memory)** deep learning model to forecast short-term cryptocurrency prices. It integrates real-time data from the **CoinGecko API**, processes features like **price, market cap, developer activity**, and outputs **predicted prices**, **risk scores**, and **investment recommendations**.

---

## 🚀 Features

- 📈 **LSTM model** trained on historical crypto market data
- 🔗 Integrated with **CoinGecko** for real-time price, volume, and market cap
- 📊 Predicts the **next 24-hour price** of selected coins
- ✅ Calculates **confidence intervals**, **risk levels**, and **investment sentiment**
- 🧠 Generates **natural-language insights** to help understand trends
- 🌍 Supports any coin available on CoinGecko by `coinId` (e.g., `bitcoin`, `ethereum`, etc.)

---

## 📦 Tech Stack

- **Flask** (API)
- **TensorFlow/Keras** (LSTM deep learning model)
- **Pandas**, **NumPy**, **Scikit-learn**
- **CoinGecko API**
- **CORS** (Cross-Origin support)
- **Logging** for monitoring


