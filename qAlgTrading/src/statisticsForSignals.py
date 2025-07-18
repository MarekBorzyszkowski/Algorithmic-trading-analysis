import json
import sys

import numpy as np
import pandas as pd

from qAlgTrading.algorithms.LinearRegressionAlgorithm import LinearRegressionAlgorithm
from qAlgTrading.algorithms.ModelsConsts import CLASSIFIERS
from qAlgTrading.algorithms.PcaAlgorithm import PcaAlgorithm
from qAlgTrading.algorithms.PcaRegAlgorithm import PcaRegAlgorithm
from qAlgTrading.algorithms.QPcaAlgorithm import QPcaAlgorithm
from qAlgTrading.algorithms.QPcaRegAlgorithm import QPcaRegAlgorithm
from qAlgTrading.algorithms.QSvcAlgorithm import QSvcAlgorithm
from qAlgTrading.algorithms.QSvrAlgorithm import QSvrAlgorithm
from qAlgTrading.algorithms.SIGNALS_CONSTS import KEEP, BUY, SELL, SIGNALS
from qAlgTrading.algorithms.SvcAlgorithm import SvcAlgorithm
from qAlgTrading.algorithms.SvrAlgorithm import SvrAlgorithm


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

is_classifier = use_svc or use_qsvc or ((use_pca or use_qpca) and (selected_model in CLASSIFIERS))

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
    else LinearRegressionAlgorithm() if use_lr else None
print(f"{algorithm.name()} initialized")
print("End of initialization")
algorithm_name = algorithm.name()

train_data = filtered_data.iloc[:int(train_data_percent * len(filtered_data))]
test_data = filtered_data.iloc[int(train_data_percent * len(filtered_data)):]

test_data_close = test_data.iloc[5:]['Close'].values
prediction_dates = test_data.iloc[5:]['Date'].values
signals_tests = [int(a) for a in _prepare_signals(test_data['Close'].values)]

results_file = {'Dates': list(prediction_dates)}
test_predictions = {'Test Data': list(test_data_close)}
test_predictions['Test signals'] = list(signals_tests)

with open(f"{newpath}/info/{algorithm_name}_predictions_results.json", "r") as file:
    json_output = json.load(file)
with open(f"{newpath}/results/{algorithm_name}_predictions.json", "r") as file:
    predictions_results = json.load(file)['predictions']

predicted_signals = predictions_results[algorithm_name] if is_classifier else predictions_results['signals']

results_stats = {'TEST_BUY':0,
                'TEST_SELL':0,
                'TEST_KEEP':0,
                'TEST_BUY_PRED_BUY':0,
                'TEST_SELL_PRED_SELL':0,
                'TEST_KEEP_PRED_KEEP':0,
                'TEST_SELL_PRED_BUY':0,
                'TEST_SELL_PRED_KEEP':0,
                'TEST_BUY_PRED_SELL':0,
                'TEST_BUY_PRED_KEEP':0,
                'TEST_KEEP_PRED_BUY':0,
                'TEST_KEEP_PRED_SELL':0,
                 'ALL_SIGNALS':0
                 }

for i in range(len(predicted_signals)):
    test_signal = SIGNALS[signals_tests[i]]
    predicted_signal = SIGNALS[predicted_signals[i]]
    results_stats[f'TEST_{test_signal}'] += 1
    results_stats[f'TEST_{test_signal}_PRED_{predicted_signal}'] += 1
    results_stats['ALL_SIGNALS'] += 1


for signal in SIGNALS.values():
    if results_stats[f'TEST_{signal}'] == 0:
        continue
    results_stats[f'{signal}_GOOD_PERCENT'] = (results_stats[f'TEST_{signal}_PRED_{signal}']  /
                                               results_stats[f'TEST_{signal}'] * 100)
    results_stats[f'{signal}_BAD_PERCENT'] = ((results_stats[f'TEST_{signal}'] - results_stats[f'TEST_{signal}_PRED_{signal}']) /
                                               results_stats[f'TEST_{signal}'] * 100)
results_stats['CORRECT_COUNT'] = results_stats['TEST_BUY_PRED_BUY'] + results_stats['TEST_SELL_PRED_SELL'] + results_stats['TEST_KEEP_PRED_KEEP']
results_stats['CORRECT_PERCENT'] = results_stats['CORRECT_COUNT'] / results_stats['ALL_SIGNALS']
results_stats['BAD_COUNT'] = results_stats['ALL_SIGNALS'] - results_stats['CORRECT_COUNT']
results_stats['BAD_PERCENT'] = results_stats['BAD_COUNT'] / results_stats['ALL_SIGNALS']

json_output['classification_stats'] = results_stats

with open(f"{newpath}/info/{algorithm_name}_predictions_results.json", "w") as file:
    json.dump(json_output, file, indent=4)
print("Predictions finished")
