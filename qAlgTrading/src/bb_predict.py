import json
import os
import sys
import time

import pandas as pd

from qAlgTrading.algorithms.BollingerBands import BollingerBands
from qAlgTrading.testingEnviroment.PredictionPerformer import PredictionPerformer
from qAlgTrading.testingEnviroment.ResultsPresenter import ResultPresenter

json_file_name = sys.argv[1]

json_output = {}

with open(json_file_name, "r") as file:
    loaded_data = json.load(file)

present_name = loaded_data["present_name"]
start_date = loaded_data["start_date"]
end_date = loaded_data["end_date"]
train_data_percent = loaded_data["train_data_percent"]

is_component_of_index = loaded_data["is_component_of_index"]
component = loaded_data["component"]
index = loaded_data["index"]

component_name = f"{index}_{component}"
json_output['component_name'] = component_name
json_output['start_date'] = start_date
json_output['end_date'] = end_date
json_output['train_data_percent'] = train_data_percent
newpath = f"../results/{component_name}"

print(f"{component} from {index} starts")

if is_component_of_index:
    file_path = f'../data/{index}/components/{component}.csv'
else:
    file_path = f'../data/{index}/{component}.csv'
data = pd.read_csv(file_path)
filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

print("Start of algorithm initialization")
algorithm = BollingerBands()
print(f"{algorithm.name()} initialized")
print("End of initialization")
if not os.path.exists(f"{newpath}/figures/traders/{algorithm.name()}"):
    os.makedirs(f"{newpath}/figures/traders/{algorithm.name()}")
train_data = filtered_data.iloc[:int(train_data_percent * len(filtered_data))]
test_data = filtered_data.iloc[int(train_data_percent * len(filtered_data)):]
json_output['train_data_size'] = len(train_data)
json_output['test_data_size'] = len(test_data)
json_output['begin_train_date'] = train_data.iloc[0]['Date']
json_output['end_train_date'] = train_data.iloc[-1]['Date']
json_output['begin_test_date'] = test_data.iloc[0]['Date']
json_output['end_test_date'] = test_data.iloc[-1]['Date']

algorithm_prediction_performer = PredictionPerformer()

prediction_dates = test_data['Date'].values


results_file = {'Dates': list(prediction_dates)}
results_diff = {}
results_relative_diff = {}
results_squared_diff = {}

print(f"Start prediction of {algorithm.name()}")
start = time.perf_counter()
algorithm_result = algorithm.fit(test_data)
end = time.perf_counter()

algorithm_result['BUY_SIGNAL'] = (algorithm_result['Close'] > algorithm_result['SMA']) & (
            algorithm_result['Close'].shift(1) <= algorithm_result['SMA'].shift(1))
algorithm_result['SELL_SIGNAL'] = (algorithm_result['Close'] < algorithm_result['SMA']) & (
            algorithm_result['Close'].shift(1) >= algorithm_result['SMA'].shift(1))
algorithm_name = algorithm.name()
algorithm_present_name = algorithm.pl_name()
results_presenter = ResultPresenter()
results_presenter.plot_bb(algorithm_result, title=f"{present_name} - {algorithm_present_name}",
                          component_name=component_name, with_save=True, present_name_component=present_name)
predictions = {algorithm_name : list(algorithm_result['TRADE_SIGNAL'])}
json_output['prediction_time_seconds'] = end - start
with open(f"{newpath}/info/{algorithm_name}_predictions_results.json", "w") as file:
    json.dump(json_output, file, indent=4)
results_file['predictions'] = predictions
with open(f"{newpath}/results/{algorithm_name}_predictions.json", "w") as file:
    json.dump(results_file, file, indent=4)
print(f"Predictions {algorithm.name()} finished")

# result_presenter = ResultPresenter()
# result_presenter.print_results_single_chart(predictions, prediction_dates, title=f"{component_name} results of algorithms", component_name=component_name, with_save=True, subfolder='predictions')
# result_presenter.print_results_separate_chart(predictions, prediction_dates, title=f"{component_name} results", component_name=component_name, with_save=True, subfolder='predictions')
# result_presenter.print_results_single_chart(results_diff, prediction_dates, title=f"{component_name} Test data and predicted data difference",
#                                             ylabel="Price difference", component_name=component_name, with_save=True, subfolder='predictions')
