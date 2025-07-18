import json
import os
import sys

import numpy as np
import pandas as pd

from qAlgTrading.algorithms.BollingerBands import BollingerBands
from qAlgTrading.algorithms.LinearRegressionAlgorithm import LinearRegressionAlgorithm
from qAlgTrading.algorithms.MACDAlgorithm import MACDAlgorithm
from qAlgTrading.algorithms.ModelsConsts import CLASSIFIERS
from qAlgTrading.algorithms.PcaAlgorithm import PcaAlgorithm
from qAlgTrading.algorithms.PcaRegAlgorithm import PcaRegAlgorithm
from qAlgTrading.algorithms.PerfectAlgorithm import PerfectAlgorithm
from qAlgTrading.algorithms.QPcaAlgorithm import QPcaAlgorithm
from qAlgTrading.algorithms.QPcaRegAlgorithm import QPcaRegAlgorithm
from qAlgTrading.algorithms.QSvcAlgorithm import QSvcAlgorithm
from qAlgTrading.algorithms.QSvrAlgorithm import QSvrAlgorithm
from qAlgTrading.algorithms.SIGNALS_CONSTS import KEEP, BUY, SELL
from qAlgTrading.algorithms.SvcAlgorithm import SvcAlgorithm
from qAlgTrading.algorithms.SvrAlgorithm import SvrAlgorithm
from qAlgTrading.testingEnviroment.ResultsPresenter import ResultPresenter
from qAlgTrading.testingEnviroment.TraderSimulator import TraderSimulator
from qAlgTrading.tradingAgent.BuyAndKeepTrader import BuyAndKeepTrader
from qAlgTrading.tradingAgent.RemainValueTrader import RemainValueTrader
from qAlgTrading.tradingAgent.WholeValueTrader import WholeValueTrader


def _prepare_signals(close_prices):
    Y = []
    for i in range(5, len(close_prices)):
        Y.append(BUY if close_prices[i] > close_prices[i - 1]
                 else SELL if close_prices[i] < close_prices[i - 1]
        else KEEP)
    return np.array(Y)


json_file_name = sys.argv[1]
selected_algorithm = sys.argv[2]
selected_model = sys.argv[3]
use_mle = sys.argv[4].lower() == 'true'

with open(json_file_name, "r") as file:
    loaded_data = json.load(file)
present_name = loaded_data["present_name"]

start_date = loaded_data["start_date"]
end_date = loaded_data["end_date"]
train_data_percent = loaded_data["train_data_percent"]

is_component_of_index = loaded_data["is_component_of_index"]
component = loaded_data["component"]
index = loaded_data["index"]

use_pca_reg = selected_algorithm == "pca_reg"
use_pca = selected_algorithm == "pca"
use_svr = selected_algorithm == "svr"
use_svc = selected_algorithm == "svc"
use_qpca_reg = selected_algorithm == "qpca_reg"
use_qpca = selected_algorithm == "qpca"
use_qsvr = selected_algorithm == "qsvr"
use_qsvc = selected_algorithm == "qsvc"
use_lr = selected_algorithm == "lr"
use_perfect = selected_algorithm == "perfect"
use_macd = selected_algorithm == "macd"
use_bb = selected_algorithm == "bb"

is_classifier = use_macd or use_bb or use_svc or use_qsvc or ((use_pca or use_qpca) and (selected_model in CLASSIFIERS))

component_name = f"{index}_{component}"
newpath = f"../results/{component_name}"

component_model_to_load = loaded_data["loaded_model_path"]  # component_name
loadedModelPath = f"../results/{component_model_to_load}/model"

print(f"{component} from {index} starts")

if is_component_of_index:
    file_path = f'../data/{index}/components/{component}.csv'
else:
    file_path = f'../data/{index}/{component}.csv'
data = pd.read_csv(file_path)
filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

print("Start of algorithm initialization")
algorithm = PcaRegAlgorithm() if use_pca_reg \
    else QPcaRegAlgorithm() if use_qpca_reg \
    else SvrAlgorithm() if use_svr \
    else QSvrAlgorithm() if use_qsvr \
    else PcaAlgorithm(model_selected=selected_model, use_mle=use_mle) if use_pca \
    else QPcaAlgorithm(model_selected=selected_model) if use_qpca \
    else SvcAlgorithm() if use_svc \
    else QSvcAlgorithm() if use_qsvc \
    else LinearRegressionAlgorithm() if use_lr \
    else MACDAlgorithm() if use_macd \
    else BollingerBands() if use_bb \
    else PerfectAlgorithm() if use_perfect else None
print(f"{algorithm.name()} initialized")
print("End of initialization")
algorithm_name = algorithm.name()
algorithm_present_name = algorithm.name_no_mle() if use_pca and use_mle \
    else algorithm.pl_name() if use_bb or use_lr or use_perfect \
    else algorithm.name()

train_data = filtered_data.iloc[:int(train_data_percent * len(filtered_data))]
test_data = filtered_data.iloc[int(train_data_percent * len(filtered_data)):]

test_data_close = test_data.iloc[4:-1]['Close'].values
prediction_dates = test_data.iloc[5:]['Date'].values
signals_tests = [int(a) for a in _prepare_signals(test_data['Close'].values)]

results_file = {'Dates': list(prediction_dates)}
test_predictions = {'Test Data': list(test_data_close)}
test_predictions['Test signals'] = list(signals_tests)

if use_perfect:
    predicted_signals = signals_tests
else:
    with open(f"{newpath}/results/{algorithm_name}_predictions.json", "r") as file:
        predictions_results = json.load(file)['predictions']
    predicted_signals = predictions_results[algorithm_name] if is_classifier else predictions_results['signals']

print("Trader initialization started")
initial_capital = 1000000
test_data_for_trader = test_data.iloc[4:-1]['Close'].values
traderSim = TraderSimulator()
traders = [WholeValueTrader(initial_capital=initial_capital, max_percentage_of_portfolio_in_one_trade=0.05),
           WholeValueTrader(initial_capital=initial_capital, max_percentage_of_portfolio_in_one_trade=0.2),
           WholeValueTrader(initial_capital=initial_capital, max_percentage_of_portfolio_in_one_trade=1),
           BuyAndKeepTrader(initial_capital=initial_capital, max_percentage_of_portfolio_in_one_trade=0.05),
           BuyAndKeepTrader(initial_capital=initial_capital, max_percentage_of_portfolio_in_one_trade=0.2),
           BuyAndKeepTrader(initial_capital=initial_capital, max_percentage_of_portfolio_in_one_trade=1),
           RemainValueTrader(initial_capital=initial_capital, max_percentage_of_portfolio_in_one_trade=0.05),
           RemainValueTrader(initial_capital=initial_capital, max_percentage_of_portfolio_in_one_trade=0.2),
           RemainValueTrader(initial_capital=initial_capital, max_percentage_of_portfolio_in_one_trade=1)]
traders_results = {}
initial_number_of_stocks = initial_capital // test_data_close[0]
initial_free_capital = initial_capital - initial_number_of_stocks * test_data_close[0]
test_trader_value = test_data_close * initial_number_of_stocks + initial_free_capital
treder_results_for_presentation = {present_name: test_trader_value}
traders_results['final_test_value'] = test_trader_value[-1]
traders_results['metadata'] = {}
traders_results['values'] = {}
print("Trader initialization ended")
if not os.path.exists(f"{newpath}/figures/traders/{algorithm.name()}"):
    os.makedirs(f"{newpath}/figures/traders/{algorithm.name()}")
result_presenter = ResultPresenter()
print("Trader started")
for trader in traders:
    trader_results = traderSim.performSimulationBySignal(trader, predicted_signals, test_data_for_trader)
    traders_results['metadata'][trader.name()] = {'final_trader_value': trader_results['trader_value'][-1],
                                                  'final_trader_value_precent_change': (trader_results['trader_value'][-1]/initial_capital - 1)*100,
                                                  'final_trader_value_precent_change_to_index': (trader_results['trader_value'][-1]/traders_results['final_test_value'] - 1)*100,
                                                  'min_trader_value': np.min(trader_results['trader_value']),
                                                  'max_trader_value': np.max(trader_results['trader_value']),
                                                  'trader_buy_value': trader_results['trader_buy_value'],
                                                  'trader_sell_value': trader_results['trader_sell_value'],
                                                  'trader_buy_orders_len': len(trader_results['trader_buy_orders']),
                                                  'trader_sell_orders_len': len(trader_results['trader_sell_orders'])}
    traders_results['values'][trader.name()] = {'trader_value': trader_results['trader_value'],
                                                'trader_portfolio_value': trader_results['trader_portfolio_value'],
                                                'trader_capital_value': trader_results['trader_capital_value'],
                                                'trader_buy_orders': trader_results['trader_buy_orders'],
                                                'trader_sell_orders': trader_results['trader_sell_orders'],
                                                'trader_buy_sell_volume': trader_results['trader_buy_sell_volume']
                                                }
    treder_results_for_presentation[trader.nameSimple()] = trader_results['trader_value']
    result_presenter.plot_trader_with_buy_sell(test_data_for_trader,
                                               trader_results['trader_value'], prediction_dates,
                                               trader_results['trader_buy_orders'],
                                               trader_results['trader_sell_orders'],
                                               trader_results['trader_buy_sell_volume'],
                                               trader.nameSimple(), present_name=present_name,
                                               title=f"{present_name} - {algorithm_present_name} - {trader.nameSimple()}",
                                               component_name=component_name, with_save=True,
                                               subfolder=f'traders/{algorithm.name()}', test_trader_value=test_trader_value)
result_presenter.print_results_single_chart(treder_results_for_presentation, prediction_dates,
                                            title=f'{present_name} - {algorithm_present_name} porównanie wyników agentów',
                                            component_name=component_name, with_save=True, subfolder=f'traders', alpha=0.8)

with open(f"{newpath}/info/{algorithm_name}_traders_results.json", "w") as file:
    json.dump(traders_results, file, indent=4)

print(f"Trading {algorithm_name} finished")

