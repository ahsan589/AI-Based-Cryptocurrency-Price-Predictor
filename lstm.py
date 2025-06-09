from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from pycoingecko import CoinGeckoAPI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

cg = CoinGeckoAPI()

def fetch_crypto_data(coin_id, currency="usd", days=60):
    """Fetch price, market cap, 24h percentage change, and human capital."""
    try:
        # Get market data
        data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency=currency, days=days)
        coin_data = cg.get_coin_by_id(id=coin_id)
    except Exception as e:
        raise ValueError(f"Invalid coin ID: {coin_id}")

    # Extract market data
    prices = data["prices"]
    market_caps = data["market_caps"]
    total_volumes = data["total_volumes"]

    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["market_cap"] = [m[1] for m in market_caps]
    df["volume"] = [v[1] for v in total_volumes]

    # Calculate percentage price change over the last 24 hours
    df["price_change_24h"] = df["price"].pct_change(periods=24) * 100

    # Get human capital (developer activity and community growth)
    developer_activity = coin_data["developer_data"]["commit_count_4_weeks"] or 0
    community_growth = coin_data["community_data"]["twitter_followers"] or 0

    # Add human capital (normalized)
    df["human_capital"] = (developer_activity + community_growth) / 1_000_000

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.dropna(inplace=True)

    return df

def prepare_data(df, window_size=90):
    """Prepare data for LSTM with multiple features."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[["price", "market_cap", "price_change_24h", "human_capital"]])

    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(scaled_data[i:i+window_size])
        y.append(scaled_data[i+window_size][0])  # Predict price

    X, y = np.array(X), np.array(y)
    return X, y, scaler

def build_lstm_model(input_shape):
    """Build LSTM model for multiple features."""
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, input_shape=input_shape)),
        Dropout(0.2),
        LSTM(128, return_sequences=False),
        Dense(64, activation="relu"),
        Dense(1)  # Predict the price
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model

def train_and_predict(coin_id, epochs=10):
    """Train the model, make predictions, and calculate metrics."""
    df = fetch_crypto_data(coin_id)
    X, y, scaler = prepare_data(df)

    # Split data into training and validation sets
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Build and train the model
    model = build_lstm_model((X.shape[1], X.shape[2]))
    model.fit(X_train, y_train, epochs=epochs, verbose=0)

    # Predict on validation set
    y_pred = model.predict(X_val)
    
    # Inverse transform predictions and actual values
    y_pred_inv = scaler.inverse_transform(
        np.concatenate([y_pred, np.zeros((len(y_pred), 3))], axis=1)
    )[:, 0]
    
    y_true_inv = scaler.inverse_transform(
        np.concatenate([y_val.reshape(-1, 1), np.zeros((len(y_val), 3))], axis=1)
    )[:, 0]

    # Calculate metrics
    r2 = r2_score(y_true_inv, y_pred_inv)
    
    # Log metrics to terminal
    logger.info(f"Model Accuracy Metrics for {coin_id}:")
    logger.info(f"R² Score: {r2:.4f}")

    # Predict the next price
    last_data = X[-1].reshape(1, X.shape[1], X.shape[2])
    predicted = scaler.inverse_transform(
        np.concatenate([model.predict(last_data), np.zeros((1, 3))], axis=1)
    )[0][0]

    return float(predicted)

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    data = request.get_json()
    coin_id = data.get('coinId', 'dogecoin').lower()

    try:
        prediction = train_and_predict(coin_id)
        return jsonify({'prediction': prediction})
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)