import json
import os
import sys
import time

import pandas as pd

from qAlgTrading.algorithms.LinearRegressionAlgorithm import LinearRegressionAlgorithm
from qAlgTrading.algorithms.PcaAlgorithm import PcaAlgorithm
from qAlgTrading.algorithms.PcaRegAlgorithm import PcaRegAlgorithm
from qAlgTrading.algorithms.QPcaAlgorithm import QPcaAlgorithm
from qAlgTrading.algorithms.QPcaRegAlgorithm import QPcaRegAlgorithm
from qAlgTrading.algorithms.QSvcAlgorithm import QSvcAlgorithm
from qAlgTrading.algorithms.QSvrAlgorithm import QSvrAlgorithm
from qAlgTrading.algorithms.SvcAlgorithm import SvcAlgorithm
from qAlgTrading.algorithms.SvrAlgorithm import SvrAlgorithm

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

load_models = loaded_data["load_models"]
component_model_to_load = loaded_data["loaded_model_path"]  # component_name
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

print("Start of algorithm initialization")
algorithm = PcaRegAlgorithm() if use_pca_reg \
    else QPcaRegAlgorithm() if use_qpca_reg \
    else SvrAlgorithm() if use_svr \
    else QSvrAlgorithm() if use_qsvr \
    else PcaAlgorithm(model_selected=selected_model, use_mle=use_mle) if use_pca \
    else QPcaAlgorithm(load_matrix_train=True, path=f"{newpath}/model", model_selected=selected_model) if use_qpca \
    else SvcAlgorithm() if use_svc \
    else QSvcAlgorithm() if use_qsvc \
    else LinearRegressionAlgorithm() if use_lr\
    else None
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
print("Training initialized")

if not os.path.exists(f"{newpath}/figures/traders/{algorithm.name()}"):
    os.makedirs(f"{newpath}/figures/traders/{algorithm.name()}")
print(f"Start training of {algorithm.name()}")
start = time.perf_counter()
algorithm.train(train_data)
end = time.perf_counter()
print(f"{algorithm.name()} took {end - start} seconds")
algorithm.save(f"{newpath}/model")
json_output[algorithm.name()] = {"training_time_seconds": end - start}
with open(f"{newpath}/info/{algorithm.name()}_training_results.json", "w") as file:
    json.dump(json_output, file, indent=4)

print(f"Training {algorithm.name()} finished")
