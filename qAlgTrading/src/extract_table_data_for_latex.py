import json
import sys
from math import floor, log10

from qAlgTrading.algorithms.PerfectAlgorithm import PerfectAlgorithm
from qAlgTrading.algorithms import TradingAlgorithm
from qAlgTrading.algorithms.BollingerBands import BollingerBands
from qAlgTrading.algorithms.LinearRegressionAlgorithm import LinearRegressionAlgorithm
from qAlgTrading.algorithms.MACDAlgorithm import MACDAlgorithm
from qAlgTrading.algorithms.ModelsConsts import LINEAR_REG, SVR_ALG, QSVR_ALG, QSVC_ALG, SVC_ALG
from qAlgTrading.algorithms.PcaAlgorithm import PcaAlgorithm
from qAlgTrading.algorithms.QPcaAlgorithm import QPcaAlgorithm
from qAlgTrading.algorithms.QSvcAlgorithm import QSvcAlgorithm
from qAlgTrading.algorithms.QSvrAlgorithm import QSvrAlgorithm
from qAlgTrading.algorithms.SvcAlgorithm import SvcAlgorithm
from qAlgTrading.algorithms.SvrAlgorithm import SvrAlgorithm

json_file_name = sys.argv[1]

with open(json_file_name, "r") as file:
    loaded_data = json.load(file)

component = loaded_data["component"]
index = loaded_data["index"]

component_name = f"{index}_{component}"
newpath = f"../results/{component_name}"

component_model_to_load = loaded_data["loaded_model_path"]  # component_name

print(f"{component} from {index} starts")

print("Start of algorithm initialization")
algorithms = []
algorithms.append(PerfectAlgorithm())
algorithms.append(MACDAlgorithm())
algorithms.append(BollingerBands())
algorithms.append(LinearRegressionAlgorithm())
algorithms.append(SvcAlgorithm())
algorithms.append(QSvcAlgorithm())
algorithms.append(SvrAlgorithm())
algorithms.append(QSvrAlgorithm())
algorithms.append(PcaAlgorithm(model_selected=LINEAR_REG, use_mle=True))
algorithms.append(PcaAlgorithm(model_selected=SVC_ALG, use_mle=True))
algorithms.append(PcaAlgorithm(model_selected=QSVC_ALG, use_mle=True))
algorithms.append(PcaAlgorithm(model_selected=SVR_ALG, use_mle=True))
algorithms.append(PcaAlgorithm(model_selected=QSVR_ALG, use_mle=True))
algorithms.append(QPcaAlgorithm(model_selected=LINEAR_REG))
algorithms.append(QPcaAlgorithm(model_selected=SVC_ALG))
algorithms.append(QPcaAlgorithm(model_selected=QSVC_ALG))
algorithms.append(QPcaAlgorithm(model_selected=SVR_ALG))
algorithms.append(QPcaAlgorithm(model_selected=QSVR_ALG))
print("End of initialization")


def is_reg(algorithm:TradingAlgorithm) -> bool:
    return ("MACD" not in algorithm.name()
            and "BollingerBands" not in algorithm.name()
            and "SVC" not in algorithm.name()
            and "PERFECT_ALG" not in algorithm.name())

def transform_name(algorithm_name: str) -> str:
    return (algorithm_name.replace('PERFECT_ALG', 'Algorytm perfekcyjny')
            .replace("LinearRegressionAlgorithm", "RL")
            .replace("LinearRegression", "RL")
            .replace("BollingerBands", "Wstęgi Bollingera")
            .replace("LinearRegression", "RL")
            .replace("_MLE_", " $\\rightarrow$ ")
            .replace("_", " $\\rightarrow$ ")
            .replace('SVC', 'SVM'))

round_to_n = lambda x, n: round(x, 3) if x == 0 else round(x, -int(floor(log10(abs(x)))) + (n - 1)) if abs(x) < 0.1 else round(x, 3)

def extract_trader_info(trader_info):
    result = """"""
    result += f"""& ${round_to_n(trader_info['final_trader_value'], 3)}$ """
    result += f"""& ${round_to_n(trader_info['max_trader_value'], 3)}$ """
    result += f"""& ${round_to_n(trader_info['min_trader_value'], 3)}$ """
    result += f"""& ${trader_info['trader_buy_orders_len']}$ """
    result += f"""& ${trader_info['trader_sell_orders_len']}$ """
    result += f"""& ${round_to_n(trader_info['trader_buy_value'], 3)}$ """
    result += f"""& ${round_to_n(trader_info['trader_sell_value'], 3)}$ """
    result += f"""& ${round_to_n(trader_info['final_trader_value_precent_change'], 3)}\\%$ """
    result += f"""& ${round_to_n(trader_info['final_trader_value_precent_change_to_index'], 3)}\\%$ """
    result += f""" \\\\ \n  """
    return result

reg_results = """"""
sig_results = """"""
trade_results = """"""


def trade_info(traders_metadata, algorithm_name):
    result = ""
    result += "\\multirow{9}{*}{"
    result += f"""{transform_name(algorithm_name)}"""
    result += "} "
    result += "& kit 05\\% "
    result += extract_trader_info(traders_metadata['Kup i trzymaj: 5.0%'])
    result += "& kit 20\\% "
    result += extract_trader_info(traders_metadata['Kup i trzymaj: 20.0%'])
    result += "& kit 100\\% "
    result += extract_trader_info(traders_metadata['Kup i trzymaj: 100%'])
    result += "& pm 05\\% "
    result += extract_trader_info(traders_metadata['Wymiana za procent majątku: 5.0%'])
    result += "& pm 20\\% "
    result += extract_trader_info(traders_metadata['Wymiana za procent majątku: 20.0%'])
    result += "& pm 100\\% "
    result += extract_trader_info(traders_metadata['Wymiana za procent majątku: 100%'])
    result += "& pa 05\\% "
    result += extract_trader_info(traders_metadata['Wymiana za procent aktywa lub gotówki: 5.0%'])
    result += "& pa 20\\% "
    result += extract_trader_info(traders_metadata['Wymiana za procent aktywa lub gotówki: 20.0%'])
    result += "& pa 100\\% "
    result += extract_trader_info(traders_metadata['Wymiana za procent aktywa lub gotówki: 100%'])
    result += " \\hline \n"
    return result


for algorithm in algorithms:
    print(algorithm.name(), " starts")
    with open(f"{newpath}/info/{algorithm.name()}_traders_results.json", "r") as file:
        traders_results = json.load(file)
    if is_reg(algorithm):
        with open(f"{newpath}/info/{algorithm.name()}_predictions_results.json", "r") as file:
            predictions_results = json.load(file)
        reg_results += f"""{transform_name(algorithm.name())} & ${round_to_n(predictions_results['Max_absolute_error'], 3)}$ """
        reg_results += f"""& ${round_to_n(predictions_results['Min_absolute_error'], 3)}$ & ${round_to_n(predictions_results['Mean_absolute_error'], 3)}$ """
        reg_results += f"""& ${round_to_n(predictions_results['Median_absolute_error'], 3)}$ & ${round_to_n(predictions_results['Max_relative_error'], 3)}$ """
        reg_results += f"""& ${round_to_n(predictions_results['Min_relative_error'], 3)}$ & ${round_to_n(predictions_results['Mean_relative_error'], 3)}$ """
        reg_results += f"""& ${round_to_n(predictions_results['Median_relative_error'], 3)}$ & ${round_to_n(predictions_results['Mean_square_error'], 3)}$ """
        reg_results += f""" \\\\ \n \\hline \n  """
    if "MACD" not in algorithm.name() and "BollingerBands" not in algorithm.name() and "PERFECT_ALG" not in algorithm.name():
        with open(f"{newpath}/info/{algorithm.name()}_predictions_results.json", "r") as file:
            predictions_results = json.load(file)
        classification_stats = predictions_results['classification_stats']
        sig_results += f"""{transform_name(algorithm.name())} & ${classification_stats['TEST_BUY_PRED_BUY']}$ """
        sig_results += f"""& ${classification_stats['TEST_SELL_PRED_SELL']}$ & ${classification_stats['TEST_KEEP_PRED_KEEP']}$ """
        sig_results += f"""& ${round_to_n(classification_stats['BUY_GOOD_PERCENT'], 3)}\\%$ & ${round_to_n(classification_stats['SELL_GOOD_PERCENT'], 3)}\\%$ """
        sig_results += f"""& ${round_to_n(classification_stats['KEEP_GOOD_PERCENT'], 3) if classification_stats['TEST_KEEP'] != 0 else 0}\\%$ """
        sig_results += f"""& ${classification_stats['TEST_BUY']-classification_stats['TEST_BUY_PRED_BUY']}$ & ${classification_stats['TEST_SELL'] - classification_stats['TEST_SELL_PRED_SELL']}$ """
        sig_results += f"""& ${classification_stats['TEST_KEEP']-classification_stats['TEST_KEEP_PRED_KEEP']}$ & ${round_to_n(classification_stats['BUY_BAD_PERCENT'], 3)}\\%$ """
        sig_results += f"""& ${round_to_n(classification_stats['SELL_BAD_PERCENT'], 3)}\\%$ & ${round_to_n(classification_stats['KEEP_BAD_PERCENT'], 3) if classification_stats['TEST_KEEP'] != 0 else 0}\\%$ """
        sig_results += f"""& ${classification_stats['CORRECT_COUNT']}$ & ${round_to_n(classification_stats['CORRECT_PERCENT']*100, 3)}\\%$ """
        sig_results += f"""& ${classification_stats['BAD_COUNT']}$ & ${round_to_n(classification_stats['BAD_PERCENT']*100, 3)}\\%$ """
        sig_results += f""" \\\\ \n \\hline \n  """
    traders_metadata = traders_results["metadata"]
    trade_results += trade_info(traders_metadata, algorithm.name())
    print(algorithm.name(), " ends")

with open(f"{newpath}/info/latex_reg_pred.txt", "w") as file:
    file.write(reg_results.replace('.', ','))

with open(f"{newpath}/info/latex_signal_pred.txt", "w") as file:
    file.write(sig_results.replace('.', ','))

with open(f"{newpath}/info/latex_trade_stats.txt", "w") as file:
    file.write(trade_results.replace('.', ','))

