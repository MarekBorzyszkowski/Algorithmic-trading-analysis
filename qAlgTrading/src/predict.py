import json
import sys
import time

import numpy as np
import pandas as pd

from qAlgTrading.algorithms.LinearRegressionAlgorithm import LinearRegressionAlgorithm
from qAlgTrading.algorithms.PcaAlgorithm import PcaAlgorithm
from qAlgTrading.algorithms.PcaRegAlgorithm import PcaRegAlgorithm
from qAlgTrading.algorithms.QPcaAlgorithm import QPcaAlgorithm
from qAlgTrading.algorithms.QPcaRegAlgorithm import QPcaRegAlgorithm
from qAlgTrading.algorithms.QSvcAlgorithm import QSvcAlgorithm
from qAlgTrading.algorithms.QSvrAlgorithm import QSvrAlgorithm
from qAlgTrading.algorithms.SIGNALS_CONSTS import KEEP, BUY, SELL
from qAlgTrading.algorithms.SvcAlgorithm import SvcAlgorithm
from qAlgTrading.algorithms.SvrAlgorithm import SvrAlgorithm
from qAlgTrading.testingEnviroment.PredictionPerformer import PredictionPerformer


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

json_output = {}

with open(json_file_name, "r") as file:
    loaded_data = json.load(file)

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

component_name = f"{index}_{component}"
json_output['component_name'] = component_name
json_output['start_date'] = start_date
json_output['end_date'] = end_date
json_output['train_data_percent'] = train_data_percent
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
    else LinearRegressionAlgorithm() if use_lr else None
print(f"{algorithm.name()} initialized")
print("End of initialization")

train_data = filtered_data.iloc[:int(train_data_percent * len(filtered_data))]
test_data = filtered_data.iloc[int(train_data_percent * len(filtered_data)):]
json_output['train_data_size'] = len(train_data)
json_output['test_data_size'] = len(test_data)
json_output['begin_train_date'] = train_data.iloc[0]['Date']
json_output['end_train_date'] = train_data.iloc[-1]['Date']
json_output['begin_test_date'] = test_data.iloc[0]['Date']
json_output['end_test_date'] = test_data.iloc[-1]['Date']

algorithm.load(loadedModelPath)
print(f"Model {algorithm.name()} loaded")

algorithm_prediction_performer = PredictionPerformer()

test_data_close = test_data.iloc[5:]['Close'].values
prediction_dates = test_data.iloc[5:]['Date'].values
signals_tests = [int(a) for a in _prepare_signals(test_data['Close'].values)]

results_file = {'Dates': list(prediction_dates)}
predictions = {'Test Data': list(test_data_close)}
predictions['Test signals'] = list(signals_tests)
results_file['predictions'] = predictions
with open(f"{newpath}/results/test_predictions.json", "w") as file:
    json.dump(results_file, file, indent=4)

results_file = {'Dates': list(prediction_dates)}
results_diff = {}
results_relative_diff = {}
results_squared_diff = {}

print(f"Start prediction of {algorithm.name()}")
start = time.perf_counter()
algorithm_result = algorithm_prediction_performer.perform_test(algorithm, test_data)
end = time.perf_counter()
algorithm_name = algorithm.name()
predictions = {algorithm_name : list(algorithm_result)}
json_output['prediction_time_seconds'] = end - start

results_diff[algorithm_name] = test_data_close - np.array(algorithm_result)
results_relative_diff[algorithm_name] = results_diff[algorithm_name] / test_data_close
results_squared_diff[algorithm_name] = results_diff[algorithm_name] ** 2
json_output["Max_absolute_error"] = np.max(results_diff[algorithm_name])
json_output["Min_absolute_error"] = np.min(results_diff[algorithm_name])
json_output["Mean_absolute_error"] = np.mean(results_diff[algorithm_name])
json_output["Median_absolute_error"] = np.median(results_diff[algorithm_name])
json_output["Max_relative_error"] = np.max(results_relative_diff[algorithm_name])
json_output["Min_relative_error"] = np.min(results_relative_diff[algorithm_name])
json_output["Mean_relative_error"] = np.mean(results_relative_diff[algorithm_name])
json_output["Median_relative_error"] = np.median(results_relative_diff[algorithm_name])
json_output["Mean_square_error"] = np.mean(results_squared_diff[algorithm_name])
with open(f"{newpath}/info/{algorithm_name}_predictions_results.json", "w") as file:
    json.dump(json_output, file, indent=4)
results_file['predictions'] = predictions
with open(f"{newpath}/results/{algorithm_name}_predictions.json", "w") as file:
    json.dump(results_file, file, indent=4)
print("Predictions finished")

# result_presenter = ResultPresenter()
# result_presenter.print_results_single_chart(predictions, prediction_dates, title=f"{component_name} results of algorithms", component_name=component_name, with_save=True, subfolder='predictions')
# result_presenter.print_results_separate_chart(predictions, prediction_dates, title=f"{component_name} results", component_name=component_name, with_save=True, subfolder='predictions')
# result_presenter.print_results_single_chart(results_diff, prediction_dates, title=f"{component_name} Test data and predicted data difference",
#                                             ylabel="Price difference", component_name=component_name, with_save=True, subfolder='predictions')
