import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
import logging
import schedule
import time
import joblib

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
ENERGY_STOCKS: List[str] = [
    "XOM",
    "CVX",
    "COP",
    "EOG",
    "SLB",
    "MPC",
    "VLO",
    "PSX",
    "OXY",
]  # Removed PXD
PREDICTION_HORIZON: int = 30  # 30-day prediction horizon
RETRAIN_INTERVAL: int = 7  # Retrain models every 7 days


def fetch_stock_data(
    symbol: str, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    """Fetch stock data from Yahoo Finance."""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            raise ValueError(f"No data available for {symbol}")
        return df
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for the model."""
    if df.empty:
        return pd.DataFrame()

    try:
        df["Returns"] = df["Close"].pct_change()
        df["MA_5"] = df["Close"].rolling(window=5).mean()
        df["MA_20"] = df["Close"].rolling(window=20).mean()
        df["RSI"] = calculate_rsi(df["Close"])
        df["Volume_Change"] = df["Volume"].pct_change()
        df["Volatility"] = df["Returns"].rolling(window=20).std()

        df.dropna(inplace=True)
        return df[
            ["Returns", "MA_5", "MA_20", "RSI", "Volume_Change", "Volatility"]
        ]
    except Exception as e:
        logging.error(f"Error preparing features: {str(e)}")
        return pd.DataFrame()


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def prepare_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Prepare target variable."""
    df["Target"] = df["Close"].pct_change(periods=horizon).shift(-horizon)
    df.dropna(inplace=True)
    return df


def train_ensemble_model(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[List[Any], StandardScaler]:
    """Train an ensemble model combining XGBoost, Random Forest, and LightGBM."""
    if X.empty or y.empty:
        logging.error("Empty dataset for training")
        return [], StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )

    models = [
        XGBRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42
        ),
        RandomForestRegressor(n_estimators=100, random_state=42),
        LGBMRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42
        ),
    ]

    for model in models:
        model.fit(X_train_scaled, y_train)

    ensemble_predictions = np.mean(
        [model.predict(X_test_scaled) for model in models], axis=0
    )

    mse = mean_squared_error(y_test, ensemble_predictions)
    r2 = r2_score(y_test, ensemble_predictions)

    logging.info(f"Ensemble Model - MSE: {mse:.4f}, R2: {r2:.4f}")

    return models, scaler


def predict_stock(
    symbol: str, models: List[Any], scaler: StandardScaler
) -> float:
    """Make a prediction for a given stock using the trained ensemble model."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    df = fetch_stock_data(symbol, start_date, end_date)
    df = prepare_features(df)

    if df.empty:
        logging.error(f"No valid data for prediction of {symbol}")
        return 0.0

    latest_data = df.iloc[-1:]
    latest_data_scaled = pd.DataFrame(
        scaler.transform(latest_data), columns=latest_data.columns
    )

    ensemble_prediction = np.mean(
        [model.predict(latest_data_scaled) for model in models]
    )
    return ensemble_prediction[0]


def update_predictions(models: List[Any], scaler: StandardScaler) -> None:
    """Update predictions for all stocks and save results."""
    predictions = []
    for symbol in ENERGY_STOCKS:
        try:
            predicted_return = predict_stock(symbol, models, scaler)
            predictions.append((symbol, predicted_return))
            logging.info(
                f"Predicted {PREDICTION_HORIZON}-day return for {symbol}: {predicted_return:.2%}"
            )
        except Exception as e:
            logging.error(f"Error processing {symbol}: {str(e)}")

    if predictions:
        # Sort stocks by predicted return (descending order)
        top_stocks = sorted(predictions, key=lambda x: x[1], reverse=True)

        # Save results to CSV
        df = pd.DataFrame(
            top_stocks,
            columns=["Symbol", f"Predicted_{PREDICTION_HORIZON}_Day_Return"],
        )
        df["Timestamp"] = datetime.now()
        df.to_csv(
            "energy_stocks_predictions.csv",
            mode="a",
            header=not pd.io.common.file_exists(
                "energy_stocks_predictions.csv"
            ),
            index=False,
        )

        logging.info("Predictions updated and saved to CSV")
    else:
        logging.warning("No valid predictions to save")


def train_and_save_models() -> None:
    """Train models for all stocks and save them."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)  # 3 years of data

    all_features = []
    all_targets = []

    for symbol in ENERGY_STOCKS:
        try:
            df = fetch_stock_data(symbol, start_date, end_date)
            df = prepare_features(df)
            if not df.empty:
                df = prepare_target(df, PREDICTION_HORIZON)

                features = df.drop(["Target"], axis=1)
                target = df["Target"]

                all_features.append(features)
                all_targets.append(target)
        except Exception as e:
            logging.error(f"Error processing {symbol}: {str(e)}")

    if all_features and all_targets:
        X = pd.concat(all_features, axis=0)
        y = pd.concat(all_targets, axis=0)

        models, scaler = train_ensemble_model(X, y)

        # Save models and scaler
        joblib.dump(models, "ensemble_models.joblib")
        joblib.dump(scaler, "scaler.joblib")

        logging.info("Models trained and saved")
    else:
        logging.error("No valid data for training models")


def load_models() -> Tuple[List[Any], StandardScaler]:
    """Load saved models and scaler."""
    models = joblib.load("ensemble_models.joblib")
    scaler = joblib.load("scaler.joblib")
    return models, scaler


def main() -> None:
    """Main function to run the stock prediction system."""
    # Initial training
    train_and_save_models()

    # Load models
    models, scaler = load_models()

    # Schedule regular updates
    schedule.every(10).seconds.do(
        update_predictions, models=models, scaler=scaler
    )
    schedule.every(RETRAIN_INTERVAL).days.do(train_and_save_models)

    # Run scheduled tasks
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
