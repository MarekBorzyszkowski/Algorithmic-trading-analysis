import json
import os
import sys
import time

import numpy as np
import pandas as pd

from qAlgTrading.algorithms.PcaAlgorithm import PcaAlgorithm
from qAlgTrading.algorithms.QPcaAlgorithm import QPcaAlgorithm
from qAlgTrading.algorithms.QSvrAlgorithm import QSvmAlgorithm
from qAlgTrading.algorithms.SvrAlgorithm import SvmAlgorithm
from qAlgTrading.testingEnviroment.PredictionPerformer import PredictionPerformer
from qAlgTrading.testingEnviroment.ResultsPresenter import ResultPresenter
from qAlgTrading.testingEnviroment.TraderSimulator import TraderSimulator
from qAlgTrading.tradingAgent.BuyAndKeepTrader import BuyAndKeepTrader
from qAlgTrading.tradingAgent.RemainValueTrader import RemainValueTrader
from qAlgTrading.tradingAgent.WholeValueTrader import WholeValueTrader

json_file_name = sys.argv[1]
json_output = {}

with open(json_file_name, "r") as file:
    loaded_data = json.load(file)

start_date = loaded_data["start_date"]
end_date = loaded_data["end_date"]
train_data_percent = loaded_data["train_data_percent"]

is_component_of_index = loaded_data["is_component_of_index"]
component = loaded_data["component"]
index = loaded_data["index"]

use_pca = loaded_data["use_pca"]
use_svm = loaded_data["use_svm"]
use_qpca = loaded_data["use_qpca"]
use_qsvm = loaded_data["use_qsvm"]


component_name = f"{index}_{component}"
json_output['component_name'] = component_name
json_output['start_date'] = start_date
json_output['end_date'] = end_date
json_output['train_data_percent'] = train_data_percent
newpath = f"../results/{component_name}"

load_models = loaded_data["load_models"]
component_model_to_load = loaded_data["loaded_model_path"] #component_name
loadedModelPath = f"../results/{component_model_to_load}/model"

if not os.path.exists(newpath):
    os.makedirs(newpath)
if not os.path.exists(f"{newpath}/figures"):
    os.makedirs(f"{newpath}/figures")
if not os.path.exists(f"{newpath}/figures/predictions"):
    os.makedirs(f"{newpath}/figures/predictions")
if not os.path.exists(f"{newpath}/figures/traders"):
    os.makedirs(f"{newpath}/figures/traders")
if not os.path.exists(f"{newpath}/info"):
    os.makedirs(f"{newpath}/info")
if not os.path.exists(f"{newpath}/model"):
    os.makedirs(f"{newpath}/model")
if not os.path.exists(f"{newpath}/results"):
    os.makedirs(f"{newpath}/results")
print(f"{component} from {index} starts")

if is_component_of_index:
    file_path = f'../data/{index}/components/{component}.csv'
else:
    file_path = f'../data/{index}/{component}.csv'
data = pd.read_csv(file_path)
filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

algorithms = []
print("Start of algorithm initialization")
if use_pca:
    pca_algorithm = PcaAlgorithm()
    print("PCA initialized")
    algorithms.append(pca_algorithm)
if use_svm:
    svm_algorithm = SvmAlgorithm()
    print("SVM initialized")
    algorithms.append(svm_algorithm)
if use_qpca:
    qpca_algorithm = QPcaAlgorithm()
    print("QPCA initialized")
    algorithms.append(qpca_algorithm)
if use_qsvm:
    qsvm_algorithm = QSvmAlgorithm()
    print("QSVM initialized")
    algorithms.append(qsvm_algorithm)
print("End of initialization")

train_data = filtered_data.iloc[:int(train_data_percent * len(filtered_data))]
test_data = filtered_data.iloc[int(train_data_percent * len(filtered_data)):]
json_output['train_data_size'] = len(train_data)
json_output['test_data_size'] = len(test_data)
json_output['begin_train_date'] = train_data.iloc[0]['Date']
json_output['end_train_date'] = train_data.iloc[-1]['Date']
json_output['begin_test_date'] = test_data.iloc[0]['Date']
json_output['end_test_date'] = test_data.iloc[-1]['Date']
print("Training initialized")
for algorithm in algorithms:
    if not os.path.exists(f"{newpath}/figures/traders/{algorithm.name()}"):
        os.makedirs(f"{newpath}/figures/traders/{algorithm.name()}")
    start = 0
    end = 0
    if load_models:
        algorithm.load(loadedModelPath)
        print(f"Model {algorithm.name()} loaded")
    else:
        print(f"Start training of {algorithm.name()}")
        start = time.perf_counter()
        algorithm.train(train_data)
        end = time.perf_counter()
        print(f"{algorithm.name()} took {end - start} seconds")
        algorithm.save(f"{newpath}/model")
    json_output[algorithm.name()] = {"training_time_seconds": end - start}
print("Training finished")

algorithm_prediction_performer = PredictionPerformer()
algorithm_results = []

test_data_close = test_data.iloc[5:]['Close'].values
prediction_dates = test_data.iloc[5:]['Date'].values
results_file = {'Dates': list(prediction_dates)}
predictions = {'Test Data': list(test_data_close)}
results_diff = {}
results_relative_diff = {}
results_squared_diff = {}

print("Predictions started")
for algorithm in algorithms:
    print(f"Start prediction of {algorithm.name()}")
    start = time.perf_counter()
    algorithm_result = algorithm_prediction_performer.perform_test(algorithm, test_data)
    end = time.perf_counter()
    algorithm_name = algorithm.name()
    predictions[algorithm_name] = list(algorithm_result)
    json_output[algorithm_name]['prediction_time_seconds'] = end - start

    results_diff[algorithm_name] = test_data_close - np.array(algorithm_result)
    results_relative_diff[algorithm_name] = results_diff[algorithm_name] / test_data_close
    results_squared_diff[algorithm_name] = results_diff[algorithm_name] ** 2
    json_output[algorithm_name]["Max_absolute_error"] = np.max(results_diff[algorithm_name])
    json_output[algorithm_name]["Min_absolute_error"] = np.min(results_diff[algorithm_name])
    json_output[algorithm_name]["Mean_absolute_error"] = np.mean(results_diff[algorithm_name])
    json_output[algorithm_name]["Median_absolute_error"] = np.median(results_diff[algorithm_name])
    json_output[algorithm_name]["Max_relative_error"] = np.max(results_relative_diff[algorithm_name])
    json_output[algorithm_name]["Min_relative_error"] = np.min(results_relative_diff[algorithm_name])
    json_output[algorithm_name]["Mean_relative_error"] = np.mean(results_relative_diff[algorithm_name])
    json_output[algorithm_name]["Median_relative_error"] = np.median(results_relative_diff[algorithm_name])
    json_output[algorithm_name]["Mean_square_error"] = np.mean(results_squared_diff[algorithm_name])
    with open(f"{newpath}/info/training_results.json", "w") as file:
        json.dump(json_output, file, indent=4)
    results_file['predictions'] = predictions
    with open(f"{newpath}/results/predictions.json", "w") as file:
        json.dump(results_file, file, indent=4)
print("Predictions finished")

result_presenter = ResultPresenter()
result_presenter.print_results_single_chart(predictions, prediction_dates, title=f"{component_name} results of algorithms", component_name=component_name, with_save=True, subfolder='predictions')
result_presenter.print_results_separate_chart(predictions, prediction_dates, title=f"{component_name} results", component_name=component_name, with_save=True, subfolder='predictions')
result_presenter.print_results_single_chart(results_diff, prediction_dates, title=f"{component_name} Test data and predicted data difference",
                                            ylabel="Price difference", component_name=component_name, with_save=True, subfolder='predictions')

print("Trader initialization started")
initial_capital = 100000
test_data_for_trader = test_data.iloc[4:-2]['Close'].values
traderSim = TraderSimulator()
print("Trader initialization ended")
print("Training started")
for algorithm in algorithms:
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
    traders_results['values'] = {}
    for trader in traders:
        trader_results = traderSim.performSimulation(trader, predictions[algorithm.name()], test_data_for_trader)
        traders_results[trader.name()] = trader_results
        traders_results['values'][trader.name()]=trader_results['trader_value']
        result_presenter.print_results_single_chart(trader_results, prediction_dates, title=f"{component_name} using {algorithm.name()} {trader.name()} results all info",component_name=component_name, with_save=True, subfolder=f'traders/{algorithm.name()}')
        result_presenter.print_results_single_chart({'trader_value': trader_results['trader_value']}, prediction_dates, title=f"{component_name} using {algorithm.name()} {trader.name()} results",component_name=component_name, with_save=True, subfolder=f'traders/{algorithm.name()}')
    result_presenter.print_results_single_chart(traders_results['values'], prediction_dates, title=f'{component_name} using {algorithm.name()} traders results comparison',component_name=component_name, with_save=True, subfolder=f'traders')
print("Training finished")