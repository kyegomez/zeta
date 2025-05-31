import uuid
import multiprocessing
import time
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import sqlite3
from pathlib import Path
from typing import Dict, Any
import csv
from loguru import logger

# Set up logging
logger.add("stock_predictions.log", rotation="1 day")

# Global variables
UPDATE_INTERVAL: int = 60 * 60  # Update every hour
DB_NAME: str = "stock_predictions.db"
RUN_ID = uuid.uuid4().hex
CSV_FILE = Path(f"predictions_{RUN_ID}.csv")
csv_lock = multiprocessing.Lock()

# List of energy stock symbols
ENERGY_STOCKS: List[str] = [
    "XOM",
    "CVX",
    "COP",
    "EOG",
    "SLB",
    "PXD",
    "MPC",
    "VLO",
    "PSX",
    "OXY",
    "KMI",
    "WMB",
    "HAL",
    "DVN",
    "BKR",
]


def create_db() -> None:
    """Create the SQLite database and predictions table if they don't exist."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS predictions
                 (symbol TEXT, predicted_return REAL, timestamp DATETIME)"""
    )
    conn.commit()
    conn.close()


def update_db(symbol: str, predicted_return: float) -> None:
    """
    Update the database with the latest prediction for a given stock symbol.

    Args:
        symbol (str): The stock symbol.
        predicted_return (float): The predicted return value.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO predictions VALUES (?, ?, ?)",
        (symbol, predicted_return, datetime.now()),
    )
    conn.commit()
    conn.close()


def fetch_stock_data(
    symbol: str, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance.

    Args:
        symbol (str): The stock symbol.
        start_date (datetime): The start date for fetching data.
        end_date (datetime): The end date for fetching data.

    Returns:
        pd.DataFrame: The fetched stock data.
    """
    logger.info(f"Fetching stock data for {symbol}")
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for the model.

    Args:
        df (pd.DataFrame): The input DataFrame with stock data.

    Returns:
        pd.DataFrame: The DataFrame with prepared features.
    """
    df["Returns"] = df["Close"].pct_change()
    df["MA_5"] = df["Close"].rolling(window=5).mean()
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["RSI"] = calculate_rsi(df["Close"])
    df["Volume_Change"] = df["Volume"].pct_change()
    df.dropna(inplace=True)
    return df


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.

    Args:
        prices (pd.Series): The price series.
        period (int, optional): The RSI period. Defaults to 14.

    Returns:
        pd.Series: The calculated RSI values.
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def prepare_target(df: pd.DataFrame, forecast_horizon: int = 5) -> pd.DataFrame:
    """
    Prepare target variable.

    Args:
        df (pd.DataFrame): The input DataFrame with stock data.
        forecast_horizon (int, optional): The forecast horizon. Defaults to 5.

    Returns:
        pd.DataFrame: The DataFrame with prepared target variable.
    """
    df["Target"] = (
        df["Close"]
        .pct_change(periods=forecast_horizon)
        .shift(-forecast_horizon)
    )
    df.dropna(inplace=True)
    return df


def train_model(
    X: pd.DataFrame, y: pd.Series, existing_model: Optional[XGBRegressor] = None
) -> Tuple[XGBRegressor, StandardScaler]:
    """
    Train or update the XGBoost model.

    Args:
        X (pd.DataFrame): The feature DataFrame.
        y (pd.Series): The target Series.
        existing_model (Optional[XGBRegressor], optional): An existing model to update. Defaults to None.

    Returns:
        Tuple[XGBRegressor, StandardScaler]: The trained model and the fitted scaler.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if existing_model:
        model = existing_model
        model.fit(X_train_scaled, y_train, xgb_model=existing_model)
    else:
        model = XGBRegressor(
            n_estimators=1000, learning_rate=0.01, max_depth=5, random_state=42
        )
        model.fit(X_train_scaled, y_train)

    test_rmse = mean_squared_error(
        y_test, model.predict(X_test_scaled), squared=False
    )
    logger.info(f"Test RMSE: {test_rmse}")

    return model, scaler


def evaluate_stock(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    existing_model: Optional[XGBRegressor] = None,
) -> Tuple[XGBRegressor, float]:
    """
    Evaluate a stock and make predictions.

    Args:
        symbol (str): The stock symbol.
        start_date (datetime): The start date for fetching data.
        end_date (datetime): The end date for fetching data.
        existing_model (Optional[XGBRegressor], optional): An existing model to update. Defaults to None.

    Returns:
        Tuple[XGBRegressor, float]: The updated model and the predicted return.
    """
    df = fetch_stock_data(symbol, start_date, end_date)
    df = prepare_features(df)
    df = prepare_target(df)

    features = ["Returns", "MA_5", "MA_20", "RSI", "Volume_Change"]
    X = df[features]
    y = df["Target"]

    model, scaler = train_model(X, y, existing_model)

    latest_data = X.iloc[-1].values.reshape(1, -1)
    latest_data_scaled = scaler.transform(latest_data)
    predicted_return = model.predict(latest_data_scaled)[0]

    update_db(symbol, predicted_return)

    # Save prediction to CSV with metadata
    metadata = {
        "last_close": df["Close"].iloc[-1],
        "last_volume": df["Volume"].iloc[-1],
        "last_rsi": df["RSI"].iloc[-1],
        "model_version": model.get_params().get("n_estimators", "unknown"),
    }
    save_prediction_to_csv(symbol, predicted_return, metadata)

    return model, predicted_return


def process_stock(symbol: str) -> None:
    """
    Process a stock continuously, updating predictions at regular intervals.

    Args:
        symbol (str): The stock symbol to process.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)
    model = None

    while True:
        try:
            model, predicted_return = evaluate_stock(
                symbol, start_date, end_date, model
            )
            logger.info(
                f"Updated prediction for {symbol}: {predicted_return:.2%}"
            )

            # Update the start_date to fetch only new data in the next iteration
            start_date = end_date
            end_date = datetime.now()
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")

        time.sleep(UPDATE_INTERVAL)


def save_prediction_to_csv(
    symbol: str, predicted_return: float, metadata: Dict[str, Any]
) -> None:
    """
    Save a prediction to a single CSV file for the entire run.

    Args:
        symbol (str): The stock symbol.
        predicted_return (float): The predicted return value.
        metadata (Dict[str, Any]): Additional metadata to save with the prediction.
    """
    with csv_lock:
        file_exists = CSV_FILE.exists()

        with CSV_FILE.open(mode="a", newline="") as f:
            fieldnames = ["timestamp", "symbol", "predicted_return"] + list(
                metadata.keys()
            )
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "predicted_return": predicted_return,
                    **metadata,
                }
            )

    logger.info(f"Saved prediction for {symbol} to {CSV_FILE}")


def main() -> None:
    """Main function to run the stock prediction system."""
    create_db()

    logger.info(f"Starting new run with ID: {RUN_ID}")
    logger.info(f"Predictions will be saved to: {CSV_FILE}")

    with multiprocessing.Pool(processes=len(ENERGY_STOCKS)) as pool:
        pool.map(process_stock, ENERGY_STOCKS)


if __name__ == "__main__":
    main()
