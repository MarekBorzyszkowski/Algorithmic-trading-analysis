from qAlgTrading.algorithms.TradingAlgorithm import TradingAlgorithm


class PerfectAlgorithm(TradingAlgorithm):
    def train(self, historical_data):
        raise NotImplementedError

    def fit(self, historical_data):
        raise NotImplementedError

    def history(self):
        raise NotImplementedError

    def save(self, directory: str):
        raise NotImplementedError

    def load(self, directory: str):
        raise NotImplementedError

    def pl_name(self):
        return "Algorytm perfekcyjny"

    def name(self):
        return "PERFECT_ALG"
