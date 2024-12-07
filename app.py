from flask import Flask, request, render_template
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Function to preprocess data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# Function to create dataset
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Function to build LSTM model
def build_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to download stock data
def download_data(stock_ticker):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 5)  # Fetch data from the last 5 years
        df = yf.download(stock_ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        if df.empty:
            return None
        return df[['Close']]
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    stock_ticker = request.form['ticker']
    
    # Download data
    data = download_data(stock_ticker)
    if data is None:
        return render_template('index.html', error=f"Failed to retrieve data for {stock_ticker}. Please try a different ticker.")
    
    # Preprocess data
    scaled_data, scaler = preprocess_data(data.values)
    X, y = create_dataset(scaled_data)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Train model
    model = build_model()
    model.fit(X, y, epochs=20, batch_size=32, verbose=0)
    
    # Predict next day's closing price
    last_60_days = scaled_data[-60:]
    next_day_input = last_60_days.reshape(1, last_60_days.shape[0], 1)
    predicted_price_scaled = model.predict(next_day_input)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(data.index[-100:], data.values[-100:], label="Actual Prices")
    plt.plot(data.index[-1:], predicted_price, 'ro', label="Predicted Price")
    plt.title(f"{stock_ticker} Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode()
    buf.close()
    
    return render_template('index.html', plot_url=f"data:image/png;base64,{plot_url}", next_day_price=predicted_price[0][0])

if __name__ == '__main__':
    app.run(debug=True)
