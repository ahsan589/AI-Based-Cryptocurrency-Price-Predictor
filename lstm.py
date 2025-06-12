from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
from pycoingecko import CoinGeckoAPI
import logging
import math

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

cg = CoinGeckoAPI()

def fetch_crypto_data(coin_id, currency="usd", days=90):
    """Fetch price, market cap, 24h percentage change, and developer activity."""
    try:
        # Get market data
        data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency=currency, days=days)
        coin_data = cg.get_coin_by_id(id=coin_id)
    except Exception as e:
        logger.error(f"CoinGecko API error: {str(e)}")
        raise ValueError(f"Invalid coin ID or API error: {coin_id}")

    # Extract market data
    prices = data["prices"]
    market_caps = data["market_caps"]
    total_volumes = data["total_volumes"]

    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["market_cap"] = [m[1] for m in market_caps]
    df["volume"] = [v[1] for v in total_volumes]

    # Calculate percentage price change over the last 24 hours
    df["price_change_24h"] = df["price"].pct_change(periods=24) * 100

    # Get developer activity
    developer_activity = coin_data["developer_data"]["commit_count_4_weeks"] or 0

    # Normalize developer activity
    df["developer_activity"] = developer_activity / 1000

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.dropna(inplace=True)
    
    # Calculate volatility (standard deviation of last 7 daily returns)
    df['daily_return'] = df['price'].pct_change()
    df['volatility'] = df['daily_return'].rolling(window=7).std() * 100 * math.sqrt(365)  # Annualized volatility
    
    return df

def prepare_data(df, window_size=30):
    """Prepare data for LSTM with multiple features."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    features = df[["price", "market_cap", "price_change_24h", "developer_activity"]]
    scaled_data = scaler.fit_transform(features)

    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i+window_size])
        y.append(scaled_data[i+window_size][0])  # Predict price

    return np.array(X), np.array(y), scaler

def build_lstm_model(input_shape):
    """Build LSTM model for multiple features."""
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, input_shape=input_shape)),
        Dropout(0.3),
        LSTM(128, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                 loss="mse",
                 metrics=['mae'])
    return model

def calculate_confidence_interval(predictions, actuals, confidence=0.95):
    """Calculate confidence interval based on prediction errors"""
    errors = actuals - predictions
    std_error = np.std(errors)
    z_score = 1.96  # For 95% confidence
    margin_of_error = z_score * std_error
    return margin_of_error

def generate_risk_assessment(predicted_change, volatility, ci_width, r2_score):
    """Generate risk metrics and recommendations"""
    # Risk score components (0-100)
    volatility_score = min(100, max(0, volatility * 10))  # 10% volatility = 100 score
    uncertainty_score = min(100, ci_width * 200)  # 5% CI width = 100 score
    model_confidence = min(100, r2_score * 100)
    
    risk_score = (volatility_score * 0.5 + uncertainty_score * 0.3 + (100 - model_confidence) * 0.2)
    
    # Risk classification
    if risk_score <= 40:
        risk_level = "Low"
    elif risk_score <= 70:
        risk_level = "Medium"
    else:
        risk_level = "High"
    
    # Investment recommendation
    if predicted_change > 1.5:
        recommendation = "Buy"
        sentiment = "Bullish"
    elif predicted_change < -1.5:
        recommendation = "Sell"
        sentiment = "Bearish"
    else:
        recommendation = "Hold"
        sentiment = "Neutral"
        
    return {
        "risk_level": risk_level,
        "risk_score": round(float(risk_score), 2),
        "investment_recommendation": recommendation,
        "market_sentiment": sentiment
    }

def generate_insights(predicted_change, volatility, volume_change, model_accuracy):
    """Generate human-readable insights"""
    trend = "upside" if predicted_change > 0 else "downside"
    
    insights = [
        f"Short-term trend indicates {abs(predicted_change):.2f}% {trend} potential with "
        f"{'elevated' if volatility > 8 else 'moderate'} volatility",
        
        f"Model accuracy (R²={model_accuracy:.2f}) suggests "
        f"{'strong' if model_accuracy > 0.85 else 'moderate' if model_accuracy > 0.7 else 'limited'} "
        "predictive confidence for 24-hour movements",
        
        "Key immediate risks: Market reaction to Bitcoin ETF flows, regulatory announcements, "
        "and liquidity changes in derivatives markets"
    ]
    
    return insights

def train_and_predict(coin_id, epochs=5):
    """Train the model and generate comprehensive prediction output"""
    try:
        df = fetch_crypto_data(coin_id)
        if len(df) < 60:
            raise ValueError("Insufficient historical data for reliable prediction")
            
        X, y, scaler = prepare_data(df)
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        model = build_lstm_model((X.shape[1], X.shape[2]))
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        model.fit(X_train, y_train, 
                 validation_data=(X_val, y_val),
                 epochs=epochs, 
                 batch_size=16,
                 callbacks=[early_stop],
                 verbose=0)
        
        # Validation predictions
        y_pred_val = model.predict(X_val, verbose=0)
        dummy_val = np.zeros((len(y_pred_val), 4))
        dummy_val[:, 0] = y_pred_val.flatten()
        y_pred_inv = scaler.inverse_transform(dummy_val)[:, 0]
        
        dummy_true = np.zeros((len(y_val), 4))
        dummy_true[:, 0] = y_val
        y_true_inv = scaler.inverse_transform(dummy_true)[:, 0]
        
        # Model metrics
        r2 = r2_score(y_true_inv, y_pred_inv)
        mae = mean_absolute_error(y_true_inv, y_pred_inv)
        ci_margin = calculate_confidence_interval(y_pred_inv, y_true_inv)
        
        # Current market data
        current_price = df["price"].iloc[-1]
        price_24h_change = df["price_change_24h"].iloc[-1]
        volatility = df["volatility"].iloc[-1] or 5.0  # Default to 5% if NaN
        
        # Future prediction
        last_window = X[-1].reshape(1, X.shape[1], X.shape[2])
        predicted_scaled = model.predict(last_window, verbose=0)[0][0]
        dummy_pred = np.zeros((1, 4))
        dummy_pred[0, 0] = predicted_scaled
        predicted_price = scaler.inverse_transform(dummy_pred)[0][0]
        
        # Calculate predicted percentage change
        predicted_change = ((predicted_price - current_price) / current_price) * 100
        
        # Confidence interval
        ci_percentage = (ci_margin / current_price) * 100
        lower_bound = max(0, predicted_price - ci_margin)
        upper_bound = predicted_price + ci_margin
        
        # Risk assessment
        risk_data = generate_risk_assessment(
            predicted_change, 
            volatility, 
            ci_percentage, 
            r2
        )
        
        # Insights
        volume_change = (df["volume"].iloc[-1] - df["volume"].iloc[-24]) / df["volume"].iloc[-24] * 100
        insights = generate_insights(predicted_change, volatility, volume_change, r2)
        
        return {
            "cryptocurrency": coin_id.capitalize(),
            "predicted_price": predicted_price,
            "current_price": current_price,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "predicted_change": predicted_change,
            "risk_data": risk_data,
            "insights": insights,
            "model_metrics": {
                "r2_score": r2,
                "mae": mae,
                "ci_margin": ci_margin
            }
        }
        
    except Exception as e:
        logger.exception("Prediction error")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    """Enhanced prediction endpoint with comprehensive output"""
    data = request.get_json()
    coin_id = data.get('coinId', 'bitcoin').lower()
    
    try:
        result = train_and_predict(coin_id)
        
        response = {
            "cryptocurrency": result["cryptocurrency"],
            "prediction_timeframe": "24 hours",
            "predicted_price": {
                "value": round(result["predicted_price"], 4),
                "confidence_interval": {
                    "lower_bound": round(result["lower_bound"], 4),
                    "upper_bound": round(result["upper_bound"], 4)
                }
            },
            "risk_assessment": result["risk_data"],
            "prediction_insights": result["insights"],
            "disclaimer": "24-hour AI-generated prediction. Not financial advice. Cryptocurrency markets are highly volatile. Actual results may vary significantly. Use at your own risk."
        }
        
        # Log critical metrics
        logger.info(f"Prediction for {coin_id}: "
                   f"Price=${response['predicted_price']['value']:.4f} | "
                   f"Risk={response['risk_assessment']['risk_level']} | "
                   f"Recommendation={response['risk_assessment']['investment_recommendation']}")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({
            "error": "Prediction unavailable",
            "details": str(e),
            "disclaimer": "Predictions are estimates only. Consult financial professionals before trading."
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
