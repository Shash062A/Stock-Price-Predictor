# 📈 Apple Stock Price Predictor (LSTM)
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras)
![NumPy](https://img.shields.io/badge/NumPy-Scientific-blue?logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-purple?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-informational)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikitlearn)
![yfinance](https://img.shields.io/badge/yfinance-Stock%20Data-green)
![pandas--datareader](https://img.shields.io/badge/pandas--datareader-API%20Data-green)
![Alpha%20Vantage](https://img.shields.io/badge/Alpha%20Vantage-API-blue)
![LSTM](https://img.shields.io/badge/Model-LSTM-brightgreen)
![Machine Learning](https://img.shields.io/badge/Type-Time%20Series-yellow)
![dotenv](https://img.shields.io/badge/python--dotenv-Env%20Vars-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Learning%20Project-yellow)
![Requirements](https://img.shields.io/badge/requirements.txt-available-brightgreen)
![Last Commit](https://img.shields.io/github/last-commit/Shash062A/Stock-Price-Predictor)
![Stars](https://img.shields.io/github/stars/Shash062A/Stock-Price-Predictor?style=social)
![Forks](https://img.shields.io/github/forks/Shash062A/Stock-Price-Predictor?style=social)

A simple machine learning project that predicts Apple stock prices using an LSTM neural network.  
Built for **learning purposes**, not real-world trading.


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

## 🧰 Tech Stack

### 🧠 Programming & Environment
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

### 🤖 Machine Learning / Deep Learning
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-f7931e?logo=scikit-learn)

### 📊 Data Processing & Analysis
![NumPy](https://img.shields.io/badge/NumPy-Scientific-blue?logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)

### 📈 Data Visualization
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blue)

### 💹 Stock Market Data Sources
![yfinance](https://img.shields.io/badge/yfinance-Stock%20Data-green)
![pandas-datareader](https://img.shields.io/badge/pandas--datareader-API%20Data-yellow)
![Alpha Vantage](https://img.shields.io/badge/Alpha%20Vantage-API-purple)

### 🔐 Environment Management
![python-dotenv](https://img.shields.io/badge/python--dotenv-Env%20Vars-yellowgreen)



---

## 📂 Project Structure
Stock-Price-Predictor/  

├── stock_predictor.ipynb      # Main Jupyter Notebook  

├── AAPL_full_data.csv         # Cached historical stock data  

├── last_price_cache.csv       # Fallback cache for API failures  

├── requirements.txt           # Project dependencies  

├── .env                       # API keys (not included in repo)  

└── README.md                  # Project documentation

---

##📄 requirements.txt
numpy>=1.23  

pandas>=1.5  

matplotlib>=3.6  

scikit-learn>=1.2  

tensorflow>=2.10  

keras>=2.10  

yfinance>=0.2  

pandas-datareader>=0.10  

python-dotenv>=1.0

---

## 🔑 Environment Setup

1. Clone the repository
```bash
 git clone https://github.com/Shash062A/Stock-Price-Predictor.git
 cd Stock-Price-Predictor```

5. Install required libraries:
---

## 📦 Installation

Install dependencies using:

```bash
pip install numpy pandas ...
  

ALPHAVANTAGE_API_KEY=your_api_key_here

```
## bash pip install numpy pandas matplotlib scikit-learn yfinance pandas-datareader tensorflow keras python-dotenv

```

ALPHAVANTAGE_API_KEY=your_api_key_here

```

## ▶️ How It Works (High-Level)

- Fetch Apple stock price data (2019 → present)

- Extract Close prices

- Scale data to range 0–1

- Build 60-day rolling windows

- Train LSTM model

- Test predictions

- Predict next closing price

- Visualize results

```
```

## ⚠️ Known Limitations

- No hyperparameter tuning

- Minimal error handling

- Model retrains every run

- Accuracy is not optimized

- Uses simple architecture

- Not suitable for real trading

- This project is intentionally simple and imperfect — the goal is to learn, not to build a production-grade trading system.

```
```

## 📌 Future Improvements

- Add proper train/validation split

- Save & load trained model

- Improve error handling

- Add technical indicators (RSI, MACD, EMA)

- Hyperparameter tuning

- Multi-stock support

- Web interface (Streamlit or Flask)

```
```

## 📜 Disclaimer

- This project is for educational purposes only.
- It does not provide financial advice.
- Do not use this model for real trading decisions.

```
```

## 🙌 Author

- Shash
- Computer Science Engineering Student
- Learning Machine Learning & Stock Prediction
- GitHub: https://github.com/Shash062A
```
