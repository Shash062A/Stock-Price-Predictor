# 📈 Apple Stock Price Predictor (LSTM)

This project is a **learning-focused machine learning experiment** built in **Jupyter Notebook** to understand how **Long Short-Term Memory (LSTM)** networks work for time-series forecasting.

The model uses the **past 60 days of Apple (AAPL) stock prices** to predict the **next closing price**.  
It is *not* intended for real trading or financial advice — it’s purely for educational purposes.

---

## 🚀 Features

- Fetches historical stock data for **Apple Inc. (AAPL)**
- Supports multiple data sources:
  - **Stooq**
  - **Yahoo Finance (yfinance)**
  - **Alpha Vantage** (API-based fallback)
- Scales data using **MinMaxScaler**
- Trains an **LSTM neural network**
- Predicts:
  - Test-set closing prices
  - Next-day closing price
- Visualizes:
  - Training vs validation data
  - Model predictions
  - Next-day forecast

---

## 🛠️ Tech Stack

- Python  
- Jupyter Notebook  
- TensorFlow / Keras  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  
- yfinance  
- pandas-datareader  
- Alpha Vantage API (optional)  
- python-dotenv  

---

## 📂 Project Structure
├── stock_predictor.ipynb # Main Jupyter Notebook
├── AAPL_full_data.csv # Cached historical stock data
├── last_price_cache.csv # Fallback cache for API failures
├── .env # API keys (not included in repo)
└── README.md # This file


---

## 🔑 Environment Setup

1. Clone the repository  
2. Install required libraries:

```bash

pip install numpy pandas matplotlib scikit-learn yfinance pandas-datareader tensorflow keras python-dotenv

ALPHAVANTAGE_API_KEY=your_api_key_here

## ▶️ How It Works (High-Level)

Fetch Apple stock price data (2019 → present)

Extract Close prices

Scale data to range 0–1

Build 60-day rolling windows

Train LSTM model

Test predictions

Predict next closing price

Visualize results

## ⚠️ Known Limitations

No hyperparameter tuning

Minimal error handling

Model retrains every run

Accuracy is not optimized

Uses simple architecture

Not suitable for real trading

This project is intentionally simple and imperfect — the goal is to learn, not to build a production-grade trading system.

## 📌 Future Improvements

Add proper train/validation split

Save & load trained model

Improve error handling

Add technical indicators (RSI, MACD, EMA)

Hyperparameter tuning

Multi-stock support

Web interface (Streamlit or Flask)

## 📜 Disclaimer

This project is for educational purposes only.
It does not provide financial advice.
Do not use this model for real trading decisions.

## 🙌 Author

Shash
Computer Science Engineering Student
Learning Machine Learning & Stock Prediction
GitHub: your-github-username-here
