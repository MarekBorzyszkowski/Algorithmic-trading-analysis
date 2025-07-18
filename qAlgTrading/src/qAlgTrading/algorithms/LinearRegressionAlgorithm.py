import os
import pickle

import numpy as np
from sklearn.linear_model import LinearRegression

from qAlgTrading.algorithms.TradingAlgorithm import TradingAlgorithm


class LinearRegressionAlgorithm(TradingAlgorithm):
    def __init__(self, history_length=5):
        self.history_length = history_length
        self.model = LinearRegression()
        self.history_data = None

    def train(self, historical_data):
        if 'Close' not in historical_data.columns:
            raise ValueError("Historical data must contain column: 'Close'")

        close_prices = historical_data['Close'].values

        X = self._prepare_features(close_prices)
        Y = close_prices[self.history_length:]
        self.model.fit(X, Y)

        self.history_data = historical_data

    def fit(self, historical_data):
        if 'Close' not in historical_data.columns:
            raise ValueError("Recent data must contain column: 'Close'")

        if len(historical_data) < self.history_length:  # Do sprawdzenia
            raise ValueError("Insufficient data for prediction.")

        X = self._prepare_features_to_fit(historical_data['Close'].values)  # Do sprawdzenia

        return self.model.predict(X).item()

    def history(self):
        return self.history_data

    def save(self, directory: str):
        with open(os.path.join(directory, f"{self.name()}_model.pkl"), "wb") as f:
            pickle.dump(self.model, f)

    def load(self, directory: str):
        with open(os.path.join(directory, f"{self.name()}_model.pkl"), "rb") as f:
            self.model = pickle.load(f)

    def pl_name(self):
        return "Regresja liniowa"

    def name(self):
        return f"LinearRegressionAlgorithm"

    def _prepare_features(self, close_prices):
        X = []
        for i in range(len(close_prices) - self.history_length):
            X.append(close_prices[i:i + self.history_length])
        return np.array(X)

    def _prepare_features_to_fit(self, close_prices):
        X = []
        for i in range(len(close_prices)):
            X.append(close_prices[i])
        return np.array(X).reshape(-1, self.history_length)
