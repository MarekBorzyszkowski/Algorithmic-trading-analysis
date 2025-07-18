import json
import sys

import numpy as np
import pandas as pd

from qAlgTrading.algorithms.LinearRegressionAlgorithm import LinearRegressionAlgorithm
from qAlgTrading.algorithms.ModelsConsts import LINEAR_REG, SVR_ALG, QSVR_ALG
from qAlgTrading.algorithms.PcaAlgorithm import PcaAlgorithm
from qAlgTrading.algorithms.QPcaAlgorithm import QPcaAlgorithm
from qAlgTrading.algorithms.QSvrAlgorithm import QSvrAlgorithm
from qAlgTrading.algorithms.SvrAlgorithm import SvrAlgorithm
from qAlgTrading.testingEnviroment.ResultsPresenter import ResultPresenter

json_file_name = sys.argv[1]
# selected_algorithm = sys.argv[2]
# selected_model = sys.argv[3]

with open(json_file_name, "r") as file:
    loaded_data = json.load(file)

start_date = loaded_data["start_date"]
end_date = loaded_data["end_date"]
train_data_percent = loaded_data["train_data_percent"]

is_component_of_index = loaded_data["is_component_of_index"]
component = loaded_data["component"]
index = loaded_data["index"]
present_name = loaded_data["present_name"]

# use_pca_reg = selected_algorithm == "pca_reg"
# use_pca = selected_algorithm == "pca"
# use_svr = selected_algorithm == "svr"
# use_svc = selected_algorithm == "svc"
# use_qpca_reg = selected_algorithm == "qpca_reg"
# use_qpca = selected_algorithm == "qpca"
# use_qsvr = selected_algorithm == "qsvr"
# use_qsvc = selected_algorithm == "qsvc"
# use_lr = selected_algorithm == "lr"

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
algorithms = []
algorithms.append(LinearRegressionAlgorithm())
algorithms.append(SvrAlgorithm())
algorithms.append(QSvrAlgorithm())
algorithms.append(PcaAlgorithm(model_selected=LINEAR_REG, use_mle=True))
algorithms.append(PcaAlgorithm(model_selected=SVR_ALG, use_mle=True))
algorithms.append(PcaAlgorithm(model_selected=QSVR_ALG, use_mle=True))
algorithms.append(QPcaAlgorithm(model_selected=LINEAR_REG))
algorithms.append(QPcaAlgorithm(model_selected=SVR_ALG))
algorithms.append(QPcaAlgorithm(model_selected=QSVR_ALG))
print(f"Algorithms: {str.join(' ', [a.name() for a in algorithms])} initialized")

train_data = filtered_data.iloc[:int(train_data_percent * len(filtered_data))]
test_data = filtered_data.iloc[int(train_data_percent * len(filtered_data)):]

test_data_close = test_data.iloc[5:]['Close'].values
prediction_dates = test_data.iloc[5:]['Date'].values

results_diff = {}
quantum_results_diff = {}
classical_results_diff = {}
only_test_data = {present_name: list(test_data_close)}
predictions = {present_name: list(test_data_close)}
quantum_predictions = {present_name: list(test_data_close)}
classical_predictions = {present_name: list(test_data_close)}
min_val = min(test_data_close)
max_val = max(test_data_close)
min_diff = 0
max_diff = -10e6
print("Start data loading")
for algorithm in algorithms:
    with open(f"{newpath}/results/{algorithm.name()}_predictions.json", "r") as file:
        predictions_results = json.load(file)['predictions'][algorithm.name()]
    algorithm_present_name = algorithm.pl_name() if ("Bands" in algorithm.name()) or ("Linear" in algorithm.name())\
        else algorithm.name_no_mle() if ("MLE" in algorithm.name())  \
        else algorithm.name()
    predictions[algorithm_present_name] = predictions_results
    results_diff[algorithm_present_name] = test_data_close - np.array(predictions_results)
    min_val = min(min_val, min(predictions_results))
    max_val = max(max_val, max(predictions_results))
    min_diff = min(min_diff, min(results_diff[algorithm_present_name]))
    max_diff = max(max_diff, max(results_diff[algorithm_present_name]))
    if 'Q' in algorithm_present_name:
        quantum_predictions[algorithm_present_name] = predictions_results
        quantum_results_diff[algorithm_present_name] = results_diff[algorithm_present_name]
    else:
        classical_predictions[algorithm_present_name] = predictions_results
        classical_results_diff[algorithm_present_name] = results_diff[algorithm_present_name]
    print(f"{algorithm_present_name} data loaded")

min_val = min_val * 1.05 if min_val < 0 else min_val * 0.95
max_val = max_val * 0.95 if max_val < 0 else max_val * 1.05
min_diff = min_diff * 1.05 if min_diff < 0 else min_diff * 0.95
max_diff = max_diff * 0.95 if max_diff < 0 else max_diff * 1.05
print("Start generating charts")
result_presenter = ResultPresenter()
result_presenter.print_results_single_chart(only_test_data, prediction_dates, title=f"{present_name}", component_name=component_name, with_save=True, subfolder='predictions', custom_y=(min_val, max_val))
result_presenter.print_results_single_chart(predictions, prediction_dates, title=f"{present_name} - wynik algorytmów regresji", component_name=component_name, with_save=True, subfolder='predictions', alpha=0.8, custom_y=(min_val, max_val))
result_presenter.print_results_single_chart(quantum_predictions, prediction_dates, title=f"{present_name} - wynik algorytmów regresji - tylko kwantowe", component_name=component_name, with_save=True, subfolder='predictions', alpha=0.8, custom_y=(min_val, max_val))
result_presenter.print_results_single_chart(classical_predictions, prediction_dates, title=f"{present_name} - wynik algorytmów regresji - tylko klasyczne", component_name=component_name, with_save=True, subfolder='predictions', alpha=0.8, custom_y=(min_val, max_val))
result_presenter.print_results_separate_chart(predictions, prediction_dates, title=f"{present_name} - <> - wynik predykcji", component_name=component_name, with_save=True, subfolder='predictions', custom_y=(min_val, max_val))
result_presenter.print_results_single_chart(results_diff, prediction_dates, title=f"{present_name} - różnica między danymi testowymi a wynikami regresji",
                                            ylabel="Różnica bezwzględna w cenie aktywa", component_name=component_name, with_save=True, subfolder='predictions', alpha=0.8, custom_y=(min_diff, max_diff))
result_presenter.print_results_single_chart(quantum_results_diff, prediction_dates, title=f"{present_name} - różnica między danymi testowymi a wynikami regresji algorytmów kwantowych",
                                            ylabel="Różnica bezwzględna w cenie aktywa", component_name=component_name, with_save=True, subfolder='predictions', alpha=0.8, custom_y=(min_diff, max_diff))
result_presenter.print_results_single_chart(classical_results_diff, prediction_dates, title=f"{present_name} - różnica między danymi testowymi a wynikami regresji algorytmów klasycznych",
                                            ylabel="Różnica bezwzględna w cenie aktywa", component_name=component_name, with_save=True, subfolder='predictions', alpha=0.8, custom_y=(min_diff, max_diff))
result_presenter.print_results_separate_chart(results_diff, prediction_dates, title=f"{present_name} - <> - różnica między danymi testowymi a wynikami", ylabel="Różnica bezwzględna w cenie aktywa",
                                              component_name=component_name, with_save=True, subfolder='predictions')

