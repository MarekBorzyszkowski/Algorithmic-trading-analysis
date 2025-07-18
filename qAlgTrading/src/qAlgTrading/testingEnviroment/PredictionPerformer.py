from tqdm import tqdm

from qAlgTrading.testingEnviroment.Constants import FEATURES


class PredictionPerformer:
    def __init__(self, history_length=5):
        self.history_length = history_length

    def perform_test(self, algorithm, data):
        """
        Wykonuje test na zadanym algorytmie przy użyciu dostarczonych danych.

        :param algorithm: Obiekt będący instancją klasy dziedziczącej po TradingAlgorithm.
        :param data: Tablica dwuwymiarowa (numpy array) z wartościami akcji dla danego przedziału czasu.
                     Wiersze reprezentują różne dni, a kolumny różne aktywa.
        :return: Wyniki modelu w postaci tabeli (numpy array) zawierającej predykcje algorytmu.
        """
        predictions = []
        for i in tqdm(range(self.history_length, len(data))):
            past_five_days = data.iloc[i - self.history_length:i][FEATURES]
            predicted_close = algorithm.fit(past_five_days)
            predictions.append(predicted_close)
        return predictions
