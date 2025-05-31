import numpy as np
import pandas as pd
from typing import List, Dict, Any
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import time


class RFRealTimeDetector:
    """
    A class for simulated real-time detection of devices on radio frequencies using XGBoost and recorded data.
    """

    def __init__(self, model_path: str = "rf_model.joblib"):
        """
        Initialize the RFRealTimeDetector.

        Args:
            model_path (str): Path to save/load the trained model.
        """
        self.model_path = model_path
        self.model: XGBClassifier = None
        self.scaler: StandardScaler = None
        self.feature_names: List[str] = ["I", "Q"]
        self.data = self._generate_synthetic_data()
        self.data_index = 0

    def _generate_synthetic_data(self) -> np.ndarray:
        """
        Generate synthetic RF data for demonstration.

        Returns:
            np.ndarray: Generated synthetic RF data.
        """
        num_samples = 10000
        time = np.linspace(0, 10, num_samples)
        signal1 = np.sin(2 * np.pi * 1 * time)  # 1 Hz sine wave
        signal2 = np.sin(2 * np.pi * 2 * time)  # 2 Hz sine wave
        noise = np.random.normal(0, 0.1, num_samples)

        # Combine signals and add noise
        combined_signal = signal1 + signal2 + noise

        # Create complex signal (I/Q data)
        complex_signal = combined_signal + 1j * np.roll(combined_signal, 100)

        return complex_signal.reshape(-1, 1)

    def collect_real_time_data(self) -> Dict[str, Any]:
        """
        Simulate collecting real-time RF data by reading from the synthetic dataset.

        Returns:
            Dict[str, Any]: A dictionary containing RF data.
        """
        if self.data_index >= len(self.data):
            self.data_index = 0  # Reset to beginning if we've reached the end

        sample = self.data[self.data_index]
        self.data_index += 1

        # Extract I and Q components
        I = np.real(sample).mean()
        Q = np.imag(sample).mean()

        return {"I": I, "Q": Q}

    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess the input data.

        Args:
            data (pd.DataFrame): Input data containing features.

        Returns:
            np.ndarray: Preprocessed features.
        """
        if self.scaler is None:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(data)
        return self.scaler.transform(data)

    def train_model(self, data: pd.DataFrame) -> None:
        """
        Train the XGBoost model.

        Args:
            data (pd.DataFrame): Training data containing features and target.
        """
        X = data.drop("target", axis=1)
        y = data["target"]

        X_scaled = self.preprocess_data(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        self.model = XGBClassifier(
            learning_rate=0.1, max_depth=5, n_estimators=100, random_state=42
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.2f}")

        joblib.dump((self.model, self.scaler), self.model_path)

    def load_model(self) -> None:
        """
        Load a previously trained model and scaler.
        """
        self.model, self.scaler = joblib.load(self.model_path)

    def predict_real_time(self, data: Dict[str, Any]) -> int:
        """
        Make real-time predictions on new data.

        Args:
            data (Dict[str, Any]): Dictionary containing feature values.

        Returns:
            int: Predicted class (modulation type or other classification target).
        """
        if self.model is None:
            raise ValueError(
                "Model not trained or loaded. Call train_model() or load_model() first."
            )

        df = pd.DataFrame([data])
        X = self.preprocess_data(df)
        prediction = self.model.predict(X)[0]
        return int(prediction)

    def run_real_time_detection(
        self, duration: int = 60, interval: float = 0.5
    ) -> None:
        """
        Run real-time detection for a specified duration.

        Args:
            duration (int): Duration to run the detection in seconds.
            interval (float): Interval between each detection in seconds.
        """
        if self.model is None:
            raise ValueError(
                "Model not trained or loaded. Call train_model() or load_model() first."
            )

        start_time = time.time()
        while time.time() - start_time < duration:
            rf_data = self.collect_real_time_data()
            prediction = self.predict_real_time(rf_data)
            print(
                f"Time: {time.time() - start_time:.2f}s, Data: {rf_data}, Prediction: {prediction}"
            )
            time.sleep(interval)


# Example usage
if __name__ == "__main__":
    detector = RFRealTimeDetector()

    # Generate training data from the synthetic dataset
    print("Generating training data from the synthetic dataset...")
    training_data = []
    for _ in range(1000):
        sample = detector.collect_real_time_data()
        sample["target"] = np.random.randint(
            0, 2
        )  # Binary classification for simplicity
        training_data.append(sample)

    # Convert to DataFrame
    df = pd.DataFrame(training_data)

    # Train the model
    detector.train_model(df)

    # Run real-time detection for 30 seconds
    print("Running simulated real-time detection...")
    detector.run_real_time_detection(duration=30, interval=1)
