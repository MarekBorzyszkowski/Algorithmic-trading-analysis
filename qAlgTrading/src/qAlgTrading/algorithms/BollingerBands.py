import numpy as np
import pandas

from qAlgTrading.algorithms.SIGNALS_CONSTS import BUY, SELL, KEEP
from qAlgTrading.algorithms.TradingAlgorithm import TradingAlgorithm


class BollingerBands(TradingAlgorithm):
    def train(self, historical_data):
        pass

    def fit(self, historical_data):
        historical_data['SMA'] = historical_data['Close'].rolling(window=20).mean()
        historical_data['SD'] = historical_data['Close'].rolling(window=20).std()
        historical_data['UB'] = historical_data['SMA'] + 2 * historical_data['SD']
        historical_data['LB'] = historical_data['SMA'] - 2 * historical_data['SD']
        return self.get_bb_signals(historical_data)

    def get_bb_signals(self, dataframe: pandas.DataFrame):
        signals = np.where(dataframe["Close"] > dataframe["SMA"], BUY, SELL)
        signals = np.where(dataframe["SMA"].isnull(), KEEP, signals)
        signals_list = [signals[0]]
        for i in range(1, len(signals)):
            if signals[i] == signals[i - 1]:
                signals_list.append(KEEP)
            else:
                signals_list.append(signals[i])
        dataframe["TRADE_SIGNAL"] = signals_list
        return dataframe

    def history(self):
        pass

    def save(self, directory: str):
        pass

    def load(self, directory: str):
        pass

    def pl_name(self):
        return "Wstęgi Bollingera"

    def name(self):
        return "BollingerBands"
