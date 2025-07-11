import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Download stock data (Example: KLNMA.IS)
data = yf.download('KLNMA.IS', start='2019-01-01', end='2024-01-01')
data = data[['Close']]  # Use only the closing price

# 2. Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 3. Create dataset for training
window_size = 60
X, y = [], []

for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i-window_size:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# 4. Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X.shape[1], 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 5. Train model
model.fit(X, y, epochs=20, batch_size=32)

# 6. Predict and inverse scale
predictions = model.predict(X)
predicted_prices = scaler.inverse_transform(predictions)
true_prices = scaler.inverse_transform(y.reshape(-1, 1))

# 7. Plot results
plt.figure(figsize=(12,6))
plt.plot(true_prices, color='blue', label='Actual Closing Price')
plt.plot(predicted_prices, color='red', label='LSTM Prediction')
plt.title('Stock Price Prediction with LSTM (KLNMA.IS)')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend()
plt.show()
