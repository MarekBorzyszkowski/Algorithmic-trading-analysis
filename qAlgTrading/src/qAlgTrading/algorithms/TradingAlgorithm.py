import abc


class TradingAlgorithm(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'train') and
                callable(subclass.train) and
                hasattr(subclass, 'fit') and
                callable(subclass.fit) and
                hasattr(subclass, 'history') and
                callable(subclass.history) and
                hasattr(subclass, 'save') and
                callable(subclass.save) and
                hasattr(subclass, 'load') and
                callable(subclass.load) and
                hasattr(subclass, 'name') and
                callable(subclass.name)
                or NotImplemented)

    @abc.abstractmethod
    def train(self, historical_data):
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, historical_data):
        raise NotImplementedError

    @abc.abstractmethod
    def history(self):
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, directory: str):
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, directory: str):
        raise NotImplementedError

    @abc.abstractmethod
    def name(self):
        raise NotImplementedError
