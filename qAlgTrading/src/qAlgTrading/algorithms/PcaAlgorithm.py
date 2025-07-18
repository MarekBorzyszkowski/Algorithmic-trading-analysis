import os
import pickle

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from qAlgTrading.algorithms.ModelsConsts import LINEAR_REG, MODELS, CLASSIFIERS, SVM_ALGORITHMS, SVR_REGRESSORS
from qAlgTrading.algorithms.SIGNALS_CONSTS import BUY, SELL, KEEP
from qAlgTrading.algorithms.TradingAlgorithm import TradingAlgorithm


class PcaAlgorithm(TradingAlgorithm):
    def __init__(self, history_length=5, model_selected=LINEAR_REG, use_mle=False):
        self.history_length = history_length
        self.pca = PCA(n_components='mle' if use_mle else self.history_length)
        self.model = MODELS[model_selected]
        self.history_data = None
        self.model_selected = model_selected
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        self.use_mle = use_mle

    def train(self, historical_data):
        if 'Close' not in historical_data.columns:
            raise ValueError("Historical data must contain column: 'Close'")

        close_prices = historical_data['Close'].values

        X = self._prepare_features(close_prices)
        Y = self._prepare_signals(close_prices) if self.model_selected in CLASSIFIERS\
            else close_prices[self.history_length:] if self.model_selected is LINEAR_REG\
            else [[element] for element in close_prices[self.history_length:]]
        X_reduced = self.pca.fit_transform(X)
        if self.model_selected in SVM_ALGORITHMS:
            X_scaled = self.scaler_X.fit_transform(X_reduced)
        else :
            X_scaled = X_reduced

        if self.model_selected in SVR_REGRESSORS:
            Y_scaled = self.scaler_Y.fit_transform(Y)
        else:
            Y_scaled = Y

        self.model.fit(X_scaled, Y_scaled)

        self.history_data = historical_data

    def fit(self, historical_data):
        if 'Close' not in historical_data.columns:
            raise ValueError("Recent data must contain column: 'Close'")

        if len(historical_data) < self.history_length:  # Do sprawdzenia
            raise ValueError("Insufficient data for prediction.")

        X = self._prepare_features_to_fit(historical_data['Close'].values)  # Do sprawdzenia

        X_reduced = self.pca.transform(X)
        if self.model_selected in SVM_ALGORITHMS:
            X_scaled = self.scaler_X.transform(X_reduced)
        else:
            X_scaled = X_reduced

        return self.scaler_Y.inverse_transform([self.model.predict(X_scaled)]).item()\
            if self.model_selected in SVR_REGRESSORS\
            else self.model.predict(X_scaled).item()

    def history(self):
        return self.history_data

    def save(self, directory: str):
        with open(os.path.join(directory, f"{self.name()}.pkl"), "wb") as f:
            pickle.dump(self.pca, f)

        with open(os.path.join(directory, f"{self.name()}_model.pkl"), "wb") as f:
            pickle.dump(self.model, f)

        if self.model_selected in SVM_ALGORITHMS:
            with open(os.path.join(directory, f"{self.name()}_scaler_x.pkl"), "wb") as f:
                pickle.dump(self.scaler_X, f)

        if self.model_selected in SVR_REGRESSORS:
            with open(os.path.join(directory, f"{self.name()}_scaler_y.pkl"), "wb") as f:
                pickle.dump(self.scaler_Y, f)

    def load(self, directory: str):
        with open(os.path.join(directory, f"{self.name()}.pkl"), "rb") as f:
            self.pca = pickle.load(f)

        with open(os.path.join(directory, f"{self.name()}_model.pkl"), "rb") as f:
            self.model = pickle.load(f)

        if self.model_selected in SVM_ALGORITHMS:
            with open(os.path.join(directory, f"{self.name()}_scaler_x.pkl"), "rb") as f:
                self.scaler_X = pickle.load(f)

        if self.model_selected in SVR_REGRESSORS:
            with open(os.path.join(directory, f"{self.name()}_scaler_y.pkl"), "rb") as f:
                self.scaler_Y = pickle.load(f)

    def pl_name(self):
        return f"PCA Regresja_liniowa"

    def name(self):
        return f"PCA{'_MLE' if self.use_mle else ''}_{self.model_selected}"

    def name_no_mle(self):
        return f"PCA {self.model_selected}"

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
            Y.append(BUY if close_prices[i] > close_prices[i - 1]
                     else SELL if close_prices[i] < close_prices[i - 1]
                     else KEEP)
        return np.array(Y)
