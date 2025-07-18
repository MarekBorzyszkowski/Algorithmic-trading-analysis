from qAlgTrading.algorithms.SIGNALS_CONSTS import BUY, SELL


class TraderSimulator:
    def __init__(self):
        pass

    def performSimulation(self, trader, predictions, currentDayValue):
        trader_value = []
        trader_portfolio_value = []
        trader_capital_value = []
        trader_value.append(trader.currentTraderValue(currentDayValue[0]))
        trader_portfolio_value.append(trader.currentPortfolioValue(currentDayValue[0]))
        trader_capital_value.append(trader.currentCapitalValue())
        for i in range(len(currentDayValue)):
            trader.actUponPrediction(currentDayValue[i], predictions[i])
            trader_value.append(trader.currentTraderValue(currentDayValue[i]))
            trader_portfolio_value.append(trader.currentPortfolioValue(currentDayValue[i]))
            trader_capital_value.append(trader.currentCapitalValue())
        return {"trader_value": trader_value,"trader_portfolio_value": trader_portfolio_value,"trader_capital_value": trader_capital_value}

    def performSimulationBySignal(self, trader, signals, currentDayValue):
        trader_value = []
        trader_portfolio_value = []
        trader_capital_value = []
        trader_buy_value = 0.0
        trader_sell_value = 0.0
        trader_buy_orders = []
        trader_sell_orders = []
        trader_buy_sell_volume = []
        # trader_value.append(trader.currentTraderValue(currentDayValue[0]))
        # trader_portfolio_value.append(trader.currentPortfolioValue(currentDayValue[0]))
        # trader_capital_value.append(trader.currentCapitalValue())
        for i in range(len(currentDayValue)):
            trade_value = trader.actUponSignal(currentDayValue[i], signals[i])
            if signals[i] == BUY:
                trader_buy_value += trade_value
                trader_buy_sell_volume.append(trade_value)
                if trade_value > 0:
                    trader_buy_orders.append(i)
            elif signals[i] == SELL:
                trader_sell_value += trade_value
                trader_buy_sell_volume.append(-trade_value)
                if trade_value > 0:
                    trader_sell_orders.append(i)
            else:
                trader_buy_sell_volume.append(0)
            trader_value.append(trader.currentTraderValue(currentDayValue[i]))
            trader_portfolio_value.append(trader.currentPortfolioValue(currentDayValue[i]))
            trader_capital_value.append(trader.currentCapitalValue())
        return {"trader_value": trader_value,
                "trader_portfolio_value": trader_portfolio_value,
                "trader_capital_value": trader_capital_value,
                "trader_buy_value": trader_buy_value,
                "trader_sell_value": trader_sell_value,
                "trader_buy_orders": trader_buy_orders,
                "trader_sell_orders": trader_sell_orders,
                "trader_buy_sell_volume": trader_buy_sell_volume
                }
