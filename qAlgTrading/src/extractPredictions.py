import json
import sys

json_file_name = sys.argv[1]

with open(json_file_name, "r") as file:
    loaded_data = json.load(file)

prediction_dates = loaded_data["Dates"]

json_output = {}
predictions_done = loaded_data["predictions"]
for prediction in predictions_done:
    if prediction == 'QSVR' or prediction == 'QPCA_REG':
        results_file = {'Dates': list(prediction_dates)}
        algorithm_name = prediction
        predictions = {algorithm_name: list(predictions_done[prediction])}
        newpath = f"../results/sp500_^SPX"

        results_file['predictions'] = predictions
        with open(f"{newpath}/results/{algorithm_name}_predictions.json", "w") as file:
            json.dump(results_file, file, indent=4)