import numpy as np
import pandas
from qAlgTrading.algorithms.TradingAlgorithm import TradingAlgorithm

from qAlgTrading.algorithms.SIGNALS_CONSTS import KEEP, BUY, SELL


class MACDAlgorithm(TradingAlgorithm):
    def train(self, historical_data):
        pass

    def fit(self, historical_data):
        macd_df = self.get_MACD(historical_data)
        return self.get_MACD_signals(macd_df)

    def get_MACD(self, dataframe):
        dataframe['ema12'] = dataframe['Close'].ewm(span=12, adjust=False).mean()
        dataframe['ema26'] = dataframe['Close'].ewm(span=26, adjust=False).mean()
        macd = dataframe['ema12'] - dataframe['ema26']
        signal = macd.ewm(span=9, adjust=False).mean()
        dataframe["MACD"] = macd
        dataframe["MACD_SIGNAL"] = signal
        return dataframe

    def get_MACD_signals(self, dataframe: pandas.DataFrame):
        signals = np.where(dataframe["MACD"] > dataframe["MACD_SIGNAL"], BUY, SELL)
        signals = np.where(dataframe["MACD"].isnull(), KEEP, signals)
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

    def name(self):
        return "MACD"
