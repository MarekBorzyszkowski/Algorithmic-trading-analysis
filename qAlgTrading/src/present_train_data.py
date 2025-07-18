import json
import sys

import pandas as pd

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

component_name = f"{index}_{component}"
newpath = f"../results/{component_name}"

print(f"{component} from {index} starts")

if is_component_of_index:
    file_path = f'../data/{index}/components/{component}.csv'
else:
    file_path = f'../data/{index}/{component}.csv'
data = pd.read_csv(file_path)
filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

train_data = filtered_data.iloc[:int(train_data_percent * len(filtered_data))]

train_data_close = train_data['Close'].values
train_dates = train_data['Date'].values

only_train_data = {present_name: list(train_data_close)}
min_val = min(train_data_close)
max_val = max(train_data_close)
print(max_val)
min_val = min_val * 1.05 if min_val < 0 else min_val * 0.95
max_val = max_val * 0.95 if max_val < 0 else max_val * 1.05
print("Start generating charts")
result_presenter = ResultPresenter()
result_presenter.print_results_single_chart(only_train_data, train_dates, title=f"{present_name} - dane treningowe", component_name=component_name, with_save=True, subfolder='predictions', custom_y=(min_val, max_val))

