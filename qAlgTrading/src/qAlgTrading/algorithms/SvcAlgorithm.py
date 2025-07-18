import os
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qAlgTrading.algorithms.SIGNALS_CONSTS import BUY, SELL, KEEP
from qAlgTrading.algorithms.TradingAlgorithm import TradingAlgorithm


class SvcAlgorithm(TradingAlgorithm):
    def __init__(self, history_length=5, kernel='rbf'):
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        self.model = SVC(kernel=kernel)
        self.history_data = None
        self.history_length = history_length

    def train(self, historical_data):
        if 'Close' not in historical_data.columns:
            raise ValueError("Historical data must contain column: 'Close'")

        close_prices = historical_data['Close'].values

        X = self._prepare_features(close_prices)
        Y = self._prepare_signals(close_prices)

        X_scaled = self.scaler_X.fit_transform(X)

        self.model.fit(X_scaled, Y)

        self.history_data = historical_data

    def fit(self, historical_data):
        if 'Close' not in historical_data.columns:
            raise ValueError("Recent data must contain column: 'Close'")

        if len(historical_data) < self.history_length:
            raise ValueError("Insufficient data for prediction.")

        close_prices = historical_data['Close'].values
        X = self._prepare_features_to_fit(close_prices)

        X_scaled = self.scaler_X.transform(X)

        return self.model.predict(X_scaled).item()

    def history(self):
        return self.history_data

    def save(self, directory: str):
        with open(os.path.join(directory, "svc_scaler_x.pkl"), "wb") as f:
            pickle.dump(self.scaler_X, f)

        with open(os.path.join(directory, "svc_scaler_y.pkl"), "wb") as f:
            pickle.dump(self.scaler_Y, f)

        with open(os.path.join(directory, "svc_model.pkl"), "wb") as f:
            pickle.dump(self.model, f)

    def load(self, directory: str):
        with open(os.path.join(directory, "svc_scaler_x.pkl"), "rb") as f:
            self.scaler_X = pickle.load(f)

        with open(os.path.join(directory, "svc_scaler_y.pkl"), "rb") as f:
            self.scaler_Y = pickle.load(f)

        with open(os.path.join(directory, "svc_model.pkl"), "rb") as f:
            self.model = pickle.load(f)

    def name(self):
        return "SVC"

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

    def _prepare_signals(self, close_prices):
        Y = []
        for i in range(self.history_length, len(close_prices)):
            Y.append([BUY] if close_prices[i] > close_prices[i - 1]
                     else [SELL] if close_prices[i] < close_prices[i - 1]
                     else [KEEP])
        return np.array(Y)
