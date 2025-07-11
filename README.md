# Stock Price Prediction using LSTM (KLNMA.IS)

This project demonstrates how to predict stock prices using an LSTM neural network built with TensorFlow/Keras.

## Dataset

- Historical daily stock data of **KLNMA.IS**
- Source: Yahoo Finance via `yfinance` API
- Range: 2019-01-01 to 2024-01-01
- Target: `Close` (closing price)

## Steps

1. Download and clean historical price data
2. Normalize data using MinMaxScaler
3. Create a time-series dataset with a 60-day window
4. Train an LSTM model
5. Predict and inverse-transform values
6. Plot actual vs. predicted prices

## Model

- LSTM with 50 units
- Single Dense output layer
- Loss function: Mean Squared Error
- Optimizer: Adam

## How to Run

```bash
pip install yfinance numpy pandas matplotlib scikit-learn tensorflow
python stock_lstm_predictor.py
```

## Output

The model plots the true closing prices against the LSTM predictions.

