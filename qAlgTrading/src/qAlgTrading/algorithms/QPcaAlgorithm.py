import os
import pickle

import numpy as np

from qAlgTrading.algorithms.ModelsConsts import LINEAR_REG, MODELS, SVM_ALGORITHMS, SVR_REGRESSORS, CLASSIFIERS
from qAlgTrading.algorithms.SIGNALS_CONSTS import BUY, SELL, KEEP
from qAlgTrading.algorithms.TradingAlgorithm import TradingAlgorithm
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel


class QPcaAlgorithm(TradingAlgorithm):
    def __init__(self, history_length=5, kernel='rbf', load_matrix_train=False, path='', model_selected=LINEAR_REG):
        self.history_length = history_length
        self.qpca = KernelPCA(n_components=history_length, kernel='precomputed')
        self.feature_map = ZZFeatureMap(feature_dimension=history_length, reps=2, entanglement="linear")
        self.kernel = FidelityQuantumKernel()
        self.model = MODELS[model_selected]
        self.model_selected = model_selected
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        self.history_data = None
        self.train_X = None
        self.X_reduced = None
        self.matrix_train = None
        self.load_matrix_train = load_matrix_train
        if load_matrix_train:
            self.load_matrix_train_fun(path)

    def load_matrix_train_fun(self, path):
        with open(os.path.join(path, "qpca_matrix_train.pkl"), "rb") as f:
            self.matrix_train = pickle.load(f)
        print("Matrix train loaded")

    def train(self, historical_data):
        if 'Close' not in historical_data.columns:
            raise ValueError("Historical data must contain column: 'Close'")

        close_prices = historical_data['Close'].values

        X = self._prepare_features(close_prices)
        self.train_X = X
        Y = self._prepare_signals(close_prices) if self.model_selected in CLASSIFIERS\
            else close_prices[self.history_length:] if self.model_selected is LINEAR_REG\
            else [[element] for element in close_prices[self.history_length:]]
        print("Start matrix_train_prep")
        if not self.load_matrix_train:
            self.matrix_train = self.kernel.evaluate(x_vec=X)
        print("Start qpca fit transform")
        X_reduced = self.qpca.fit_transform(self.matrix_train)
        if self.model_selected in SVM_ALGORITHMS:
            X_scaled = self.scaler_X.fit_transform(X_reduced)
        else :
            X_scaled = X_reduced

        if self.model_selected in SVR_REGRESSORS:
            Y_scaled = self.scaler_Y.fit_transform(Y)
        else:
            Y_scaled = Y
        print("Start model fit")
        self.model.fit(X_scaled, Y_scaled)
        self.X_reduced = X_reduced
        self.history_data = historical_data

    def fit(self, historical_data):
        if 'Close' not in historical_data.columns:
            raise ValueError("Recent data must contain column: 'Close'")

        if len(historical_data) < self.history_length: #Do sprawdzenia
            raise ValueError("Insufficient data for prediction.")

        X = self._prepare_features_to_fit(historical_data['Close'].values) #Do sprawdzenia
        matrix_test = self.kernel.evaluate(x_vec=X, y_vec=self.train_X)
        X_reduced = self.qpca.transform(matrix_test)
        if self.model_selected in SVM_ALGORITHMS:
            X_scaled = self.scaler_X.transform(X_reduced)
        else:
            X_scaled = X_reduced

        return self.scaler_Y.inverse_transform([self.model.predict(X_scaled)]).item() \
            if self.model_selected in SVR_REGRESSORS \
            else self.model.predict(X_scaled).item()

    def history(self):
        return self.history_data

    def save(self, directory: str):
        with open(os.path.join(directory, f"{self.name()}.pkl"), "wb") as f:
            pickle.dump(self.qpca, f)

        with open(os.path.join(directory, f"{self.name()}_kernel.pkl"), "wb") as f:
            pickle.dump(self.kernel, f)

        with open(os.path.join(directory, f"{self.name()}_model.pkl"), "wb") as f:
            pickle.dump(self.model, f)

        with open(os.path.join(directory, f"{self.name()}_train_X.pkl"), "wb") as f:
            pickle.dump(self.train_X, f)

        with open(os.path.join(directory, "qpca_matrix_train.pkl"), "wb") as f:
            pickle.dump(self.matrix_train, f)

        if self.model_selected in SVM_ALGORITHMS:
            with open(os.path.join(directory, f"{self.name()}_scaler_x.pkl"), "wb") as f:
                pickle.dump(self.scaler_X, f)

        if self.model_selected in SVR_REGRESSORS:
            with open(os.path.join(directory, f"{self.name()}_scaler_y.pkl"), "wb") as f:
                pickle.dump(self.scaler_Y, f)

    def load(self, directory: str):
        with open(os.path.join(directory, f"{self.name()}.pkl"), "rb") as f:
            self.qpca = pickle.load(f)

        with open(os.path.join(directory, f"{self.name()}_kernel.pkl"), "rb") as f:
            self.kernel = pickle.load(f)

        with open(os.path.join(directory, f"{self.name()}_model.pkl"), "rb") as f:
            self.model = pickle.load(f)

        with open(os.path.join(directory, f"{self.name()}_train_X.pkl"), "rb") as f:
            self.train_X = pickle.load(f)

        with open(os.path.join(directory, "qpca_matrix_train.pkl"), "rb") as f:
            self.matrix_train = pickle.load(f)

        if self.model_selected in SVM_ALGORITHMS:
            with open(os.path.join(directory, f"{self.name()}_scaler_x.pkl"), "rb") as f:
                self.scaler_X = pickle.load(f)

        if self.model_selected in SVR_REGRESSORS:
            with open(os.path.join(directory, f"{self.name()}_scaler_y.pkl"), "rb") as f:
                self.scaler_Y = pickle.load(f)

    def pl_name(self):
        return "QPCA Regresja liniowa"

    def name(self):
        return f"QPCA_{self.model_selected}"

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
