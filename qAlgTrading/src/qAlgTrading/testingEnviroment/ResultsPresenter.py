from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas

pol_to_latin = {
    'ą': 'a',
    'ó': 'o',
    'ę': 'e',
    'ż': 'z',
    '%': 'p'
}


def replace_polish_with_latin(string: str):
    result = string
    for key, val in pol_to_latin.items():
        result = result.replace(key, val)
    return result


class ResultPresenter:
    def __init__(self):
        pass

    def print_results_single_chart(self, results, input_dates, title="Results of Algorithms", ylabel="Cena aktywa",
                                   component_name="NOT_GIVEN", with_save=False, subfolder="predictions", fig_size=(10,5), alpha=1, custom_y=None):
        """
        Prezentuje wyniki algorytmu na pojedynczym wykresie.

        :param results: Słownik, gdzie kluczem jest nazwa algorytmu, a wartością lista wyników.
                        {'Algorithm Name': [wyniki]}
        """
        dates = self.convert_dates(input_dates)
        plt.figure(figsize=fig_size)

        if custom_y is None:
            min_val = min(min([min(a) for a in results.values()]) * 1.05, 0)
            max_val = max([max(a) for a in results.values()]) * 1.05
        else:
            (min_val, max_val) = custom_y
        for name, result in results.items():
            plt.plot(dates, result, label=name.replace('SVC', 'SVM').replace('LinearRegression','Regresja liniowa').replace('_',' '), alpha=alpha)

        plt.title(title.replace('SVC', 'SVM').replace('LinearRegression','Regresja liniowa').replace('_',' '))
        plt.xlabel("Data")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.ylim(min_val, max_val)
        plt.legend()
        if with_save:
            plt.savefig(
                f"../results/{component_name}/figures/{subfolder}/{replace_polish_with_latin(title.replace(' ', '_'))}.png")
        plt.close()

    def print_results_separate_chart(self, results, input_dates, title="Results", ylabel="Cena aktywa",
                                     component_name="NOT_GIVEN", with_save=False, subfolder="predictions", custom_y=None):
        """
        Prezentuje wyniki algorytmów na osobnych wykresach.

        :param results: Słownik, gdzie kluczem jest nazwa algorytmu, a wartością lista wyników.
                        {'Algorithm Name': [wyniki]}
        """
        dates = self.convert_dates(input_dates)
        num_algorithms = len(results)

        if custom_y is None:
            min_val = min(min([min(a) for a in results.values()]) * 1.05, 0)
            max_val = max([max(a) for a in results.values()]) * 1.05
        else:
            (min_val, max_val) = custom_y
        plt.figure(figsize=(10, 5 * num_algorithms))

        for i, (name, result) in enumerate(results.items()):
            plt.subplot(num_algorithms, 1, i + 1)
            plt.plot(dates, result)
            plt.title(f"{title.replace('<>', f'{name.replace('SVC', 'SVM').replace('LinearRegression','Regresja liniowa')}').replace('_',' ')}")
            plt.xlabel("Data")
            plt.ylabel(ylabel)
            plt.ylim(min_val, max_val)
            plt.grid(True)

        plt.tight_layout()
        if with_save:
            plt.savefig(
                f"../results/{component_name}/figures/{subfolder}/{replace_polish_with_latin(title.replace(' ', '_').replace('<>', f''))}_ALL.png")
        plt.close()

        for i, (name, result) in enumerate(results.items()):
            plt.figure(figsize=(10, 5))
            plt.plot(dates, result)
            plt.title(f"{title.replace('<>', f'{name.replace('SVC', 'SVM').replace('LinearRegression','Regresja liniowa')}').replace('_',' ')}")
            plt.xlabel("Data")
            plt.ylabel(ylabel)
            plt.ylim(min_val, max_val)
            plt.grid(True)
            if with_save:
                plt.savefig(
                    f"../results/{component_name}/figures/{subfolder}/{replace_polish_with_latin(title.replace('<>', f'{name}'))}_{name}.png".replace(' ', '_'))
            plt.close()

    def plot_MACD(self, dataframe: pandas.DataFrame, title="Results of Algorithms", component_name="NOT_GIVEN",
                  with_save=False, subfolder="predictions", present_name_component="NOT_GIVEN"):
        dates = self.convert_dates(dataframe['Date'].values)
        start_index = dataframe.index[0]
        # Plot
        plt.figure(figsize=(10, 10))
        # dates[a - start_index
        # for a in dataframe.index[dataframe['Bullish_Run_Start']]]
        # Plot Close Price
        plt.subplot(2, 1, 1)
        plt.grid(True)
        plt.plot(dates, dataframe['Close'].values, label=present_name_component)
        plt.scatter(np.array(dates)[np.array(dataframe.index[dataframe['Bullish_Run_Start']]) - start_index],
                    dataframe['Close'][dataframe['Bullish_Run_Start']], marker='^', color='g',
                    label='Moment kupna')
        plt.scatter(np.array(dates)[np.array(dataframe.index[dataframe['Bearish_Run_Start']]) - start_index],
                    dataframe['Close'][dataframe['Bearish_Run_Start']], marker='v', color='r',
                    label='Moment sprzedaży')
        plt.title(title)
        plt.xlabel("Data")
        plt.ylabel("Cena aktywa")
        plt.legend()

        # Plot
        # plt.figure(figsize=(14, 10))

        # Plot MACD
        plt.subplot(2, 1, 2)
        plt.plot(dates, dataframe['MACD'].values, label='Linia MACD', color='blue',
                 alpha=0.5)
        plt.plot(dates, dataframe['MACD_SIGNAL'].values, label='Linia sygnału',
                 color='red', alpha=0.5)
        plt.xlabel("Data")
        # plt.ylabel("")
        plt.title("Wartość MACD i sygnału")
        # plt.bar(dataframe.index, dataframe['MACD_Diff'], label='Histogram', color='grey', alpha=0.5)

        # Markers for bullish and bearish crossover
        # plt.scatter(dataframe.index[dataframe['Bullish_Crossover']],
        #             dataframe['MACD'][dataframe['Bullish_Crossover']], marker='^', color='g',
        #             label='Bullish Crossover')
        # plt.scatter(dataframe.index[dataframe['Bearish_Crossover']],
        #             dataframe['MACD'][dataframe['Bearish_Crossover']], marker='v', color='r',
        #             label='Bearish Crossover')
        plt.grid(True)
        plt.legend()
        # plt.show()
        if with_save:
            plt.savefig(
                f"../results/{component_name}/figures/{subfolder}/{replace_polish_with_latin(title.replace(' ', '_'))}.png")
        plt.close()

    def plot_trader_with_buy_sell(self, index_val, results, input_dates, buy_index, sell_index,
                                  hist_vals, trader_name, present_name="NOT_GIVEN",
                                  title="Results of Algorithms", ylabel="Cena aktywa", component_name="NOT_GIVEN",
                                  with_save=False, subfolder="traders", test_trader_value=None):

        dates = self.convert_dates(input_dates)
        # Plot
        plt.figure(figsize=(10, 15))
        # Plot Close Price
        plt.subplot(3, 1, 1)
        plt.grid(True)
        plt.plot(dates, test_trader_value, label=present_name, alpha=.5, color='tab:orange')
        plt.plot(dates, results, label=trader_name, color='tab:blue')
        # plt.scatter(np.array(dates)[buy_index],
        #             np.array(results)[buy_index], marker='^', color='g',
        #             label='Moment kupna')
        # plt.scatter(np.array(dates)[sell_index],
        #             np.array(results)[sell_index], marker='v', color='r',
        #             label='Moment sprzedaży')
        plt.title(title.replace('SVC', 'SVM').replace('LinearRegression','Regresja liniowa').replace('_',' '))
        plt.xlabel("Data")
        plt.ylabel("Wartość majątku agenta")
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.grid(True)
        plt.plot(dates, index_val, label=present_name)
        plt.scatter(np.array(dates)[buy_index],
                    np.array(index_val)[buy_index], marker='^', color='g',
                    label='Moment kupna')
        plt.scatter(np.array(dates)[sell_index],
                    np.array(index_val)[sell_index], marker='v', color='r',
                    label='Moment sprzedaży')
        plt.title(f"{present_name.replace('SVC', 'SVM').replace('LinearRegression','Regresja liniowa').replace('_',' ')} z naniesionymi sygnałami kupna/sprzedaży")
        plt.xlabel("Data")
        plt.ylabel(ylabel)
        plt.legend()

        # Plot
        # plt.figure(figsize=(14, 10))

        # Plot MACD
        plt.subplot(3, 1, 3)
        plt.bar(dates, hist_vals, label='Wartość obrotu', color='grey', alpha=0.5)
        plt.title(f"Generowany obrót")
        plt.xlabel("Data")
        plt.ylabel("Wartość transakcji")
        plt.grid(True)
        plt.legend()
        # plt.show()

        if with_save:
            plt.savefig(
                f"../results/{component_name}/figures/{subfolder}/{replace_polish_with_latin(title.replace(' ', '_'))}.png")
        plt.close()


    def plot_bb(self, dataframe: pandas.DataFrame, title="Results of Algorithms", component_name="NOT_GIVEN",
                  with_save=False, subfolder="predictions", present_name_component="NOT_GIVEN"):
        """
        Prezentuje wyniki algorytmu na pojedynczym wykresie.

        :param results: Słownik, gdzie kluczem jest nazwa algorytmu, a wartością lista wyników.
                        {'Algorithm Name': [wyniki]}
        """
        dates = self.convert_dates(dataframe['Date'].values)
        start_index = dataframe.index[0]
        plt.figure(figsize=(10, 10))

        plt.subplot(2, 1, 1)
        plt.grid(True)
        plt.fill_between(dates, dataframe['UB'].values, dataframe['LB'].values, alpha=.5, linewidth=0)
        plt.plot(dates, dataframe['Close'].values, label=present_name_component)
        plt.plot(dates, dataframe['UB'].values, label="Górna wstęga", color='r')
        plt.plot(dates, dataframe['LB'].values, label="Dolna wstęga", color='g')
        plt.plot(dates, dataframe['SMA'].values, label="Środkowa wstęga", color='y')
        plt.title(title)
        plt.xlabel("Data")
        plt.ylabel("Cena aktywa")
        plt.legend()

        # Plot
        # plt.figure(figsize=(14, 10))

        # Plot MACD
        plt.subplot(2, 1, 2)
        plt.plot(dates, dataframe['Close'].values, label=present_name_component)
        plt.scatter(np.array(dates)[np.array(dataframe.index[dataframe['BUY_SIGNAL']]) - start_index],
                    dataframe['Close'][dataframe['BUY_SIGNAL']], marker='^', color='g',
                    label='Moment kupna')
        plt.scatter(np.array(dates)[np.array(dataframe.index[dataframe['SELL_SIGNAL']]) - start_index],
                    dataframe['Close'][dataframe['SELL_SIGNAL']], marker='v', color='r',
                    label='Moment sprzedaży')
        plt.xlabel("Data")
        plt.ylabel("Cena aktywa")
        plt.title("Sygnały kupna/sprzedaży")
        plt.grid(True)
        plt.legend()
        # plt.show()
        if with_save:
            plt.savefig(
                f"../results/{component_name}/figures/{subfolder}/{replace_polish_with_latin(title.replace(' ', '_'))}.png")
        plt.close()

    def convert_dates(self, dates):
        return [datetime.strptime(date, '%Y-%m-%d') for date in dates]
