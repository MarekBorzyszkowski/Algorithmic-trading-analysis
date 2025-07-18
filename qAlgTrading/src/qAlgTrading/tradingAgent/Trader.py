import abc


class Trader(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'actUponPrediction') and
                callable(subclass.actUponPrediction) and
                hasattr(subclass, 'currentPortfolioValue') and
                callable(subclass.currentPortfolioValue) and
                hasattr(subclass, 'currentTraderValue') and
                callable(subclass.currentTraderValue) and
                hasattr(subclass, 'currentCapitalValue') and
                callable(subclass.currentCapitalValue) and
                hasattr(subclass, 'name') and
                callable(subclass.name) and
                hasattr(subclass, 'nameSimple') and
                callable(subclass.nameSimple) and
                hasattr(subclass, 'actUponSignal') and
                callable(subclass.actUponSignal)
                or NotImplemented)

    @abc.abstractmethod
    def actUponPrediction(self, historical_data, prediction):
        raise NotImplemented

    @abc.abstractmethod
    def actUponSignal(self, stock_price, trading_signal):
        raise NotImplemented

    @abc.abstractmethod
    def currentPortfolioValue(self, stock_price):
        raise NotImplemented

    @abc.abstractmethod
    def currentTraderValue(self, stock_price):
        raise NotImplemented

    @abc.abstractmethod
    def currentCapitalValue(self):
        raise NotImplemented

    @abc.abstractmethod
    def name(self):
        raise NotImplemented

    @abc.abstractmethod
    def nameSimple(self):
        raise NotImplemented
