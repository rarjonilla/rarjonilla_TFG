import csv
import itertools
import numpy as np
import time
from datetime import datetime
from typing import List, Optional
from itertools import product

from sklearn.model_selection import train_test_split
from tensorflow.keras import models
from tensorflow.keras import layers

import tensorflow as tf

import pandas as pd
from pandas import DataFrame

from game_environment.game import Non_playable_game
from training.supervised.custom_csv_logger import CustomCSVLogger

# TODO -> permetre a l'usuari triar entre les funcions d'activació i les de optimització
class Supervised_training:
    """Classe Supervised Training"""

    # def __init__(self, training_type: int, game_type: int, total_games: int, num_players: int, single_mode: bool, only_assist: bool, rivals_model_type: List[int], rivals_model_name: List[str], generate_data: bool, csv_filename: str, csv_filename_2: str, prepare_data: bool, train: bool) -> None:
    def __init__(self, training_type: int, game_type: int, total_games: int, num_players: int, single_mode: bool, only_assist: bool, rivals_model_type: List[int], rivals_model_name: List[str], generate_data: bool, csv_filename: str, csv_filename_2: str, save_prepared_data: str, save_filename: str, layers: List[int] = None, do_training: bool = True) -> None:
        self.__training_type: int = training_type
        self.__game_type: int = game_type
        self.__total_games: int = total_games
        self.__num_players: int = num_players
        self.__single_mode: bool = single_mode
        self.__only_assist = only_assist
        self.__csv_filename: str = "" if csv_filename is None else csv_filename
        self.__csv_filename_2: str = "" if csv_filename_2 is None else csv_filename_2
        self.__rivals_model_type: List[int] = rivals_model_type
        self.__rivals_model_name: List[str] = rivals_model_name
        self.__df: Optional[DataFrame] = None
        self.layers: List[int] = layers

        self.__save_prepared_data: str = save_prepared_data
        self.__save_filename: str = save_filename

        prepare_data: bool = False

        if generate_data:
            print("generate")
            start_time_generate = time.time()
            self.__generate_data()
            print("--- %s seconds ---" % (time.time() - start_time_generate))

            prepare_data = True
        elif 'prepared' not in csv_filename:
            prepare_data = True

        if prepare_data:
            print("prepare")
            start_time_prepare = time.time()
            # self.__prepare_data()
            self.__prepare_data_normalized()
            print("--- %s seconds ---" % (time.time() - start_time_prepare))

        if csv_filename is not None and self.__df is None:
            print("load")
            start_time_prepare = time.time()
            self.__load_data()
            print("--- %s seconds ---" % (time.time() - start_time_prepare))

        if not prepare_data and csv_filename is None and self.__df is None:
            raise AttributeError("there is not data loaded")

        if do_training:
            # if train:
            if training_type == 4:

                # print("train round points")
                #             start_time_train = time.time()
                #             self.__training(True)
                #             print("--- %s seconds ---" % (time.time() - start_time_train))
                #
                #             print("train win or lose")
                #             start_time_train = time.time()
                #             self.__training(False)
                #             print("--- %s seconds ---" % (time.time() - start_time_train))

                print("train round points")
                start_time_train = time.time()
                self.__training(1)
                print("--- %s seconds ---" % (time.time() - start_time_train))

                print("train win or lose")
                start_time_train = time.time()
                self.__training(2)
                print("--- %s seconds ---" % (time.time() - start_time_train))

                print("train heuristics + round points")
                start_time_train = time.time()
                self.__training(3)
                print("--- %s seconds ---" % (time.time() - start_time_train))
            else:
                print("train")
                start_time_train = time.time()
                self.__training()
                print("--- %s seconds ---" % (time.time() - start_time_train))

    def __is_brisca(self) -> bool:
        return self.__game_type == 1

    def __generate_data(self) -> None:
        date = datetime.now()
        # TODO -> Primer la data, després el save_filename
        if self.__csv_filename == "":
            self.__csv_filename = "data/brisca/" if self.__is_brisca() else "data/tute"
            self.__csv_filename += "_only_assist/" if self.__only_assist else "/"
            self.__csv_filename += f"{self.__num_players}j/" if self.__single_mode else f"{self.__num_players}jt/"
            self.__csv_filename += date.strftime("%Y%m%d_%H%M%S") + "_" + self.__save_filename + ".csv"

        header = "deck_size,trump_suit_id,trump_label,"
        for i in range(1, self.__num_players):
            header += f"played_card_{i}_suit_id,"
            header += f"played_card_{i}_label,"

        for i in range(self.__num_players):
            header += f"hand_{i}_decimal_value,"

        if not self.__is_brisca():
            for i in range(self.__num_players):
                header += f"player_{i}_sing_suit_id,"

        header += "all_played_cards_decimal_value,"

        for i in range(self.__num_players):
            header += f"player_{i}_score,"

        header += "rules_decimal_value,"
        header += "actions_decimal_value,"
        header += "round_score,"
        header += "heuristics,"
        header += "win_or_lose,"

        with open(self.__csv_filename, 'a', newline='') as csv_file:
            # Crear objecte writer
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(header.split(','))

        # Es generen 8 o 16 combinacions diferents
        # Cada CSV contindrà X partides per a cada combinació
        # Si decidim 1000 partides, es generaran 1000 partides per a cada combinació (8000 o 16000)
        rules_str = "can_change last_tens black_hand"
        if self.__num_players == 2:
            rules_str += " hunt_the_three"

        # print(rules_str)
        rules_list = rules_str.split()

        for p in itertools.product([False, True], repeat=len(rules_list)):
            rules = dict(zip(rules_list, p))

            if self.__num_players > 2:
                rules['hunt_the_three'] = False

            rules['only_assist'] = self.__only_assist

            # print(rules)
            game: Non_playable_game = Non_playable_game(self.__game_type, self.__total_games, self.__rivals_model_type, self.__rivals_model_name, self.__num_players, self.__single_mode, rules, True, self.__csv_filename)

    def __dummy_vars(self, df: DataFrame, column_name: str, intervals: Optional[List[int]], labels: Optional[List[str]], all_categories: Optional[List[int]]) -> DataFrame:
        if intervals is not None and labels is not None:
            # Dividir la columna en intervals i crear variables binaries
            df_dummies = pd.get_dummies(pd.cut(df[column_name], bins=intervals, labels=labels))
        elif all_categories is not None:
            if max(all_categories) > 9:
                df[column_name] = df[column_name].apply(lambda x: '{:02d}'.format(x))

            df_dummies = pd.get_dummies(df[column_name], prefix=column_name, columns=all_categories)

            # Afegir columnes restants, si és necessari
            for category in all_categories:
                cat = '0'+str(category) if category < 10 and max(all_categories) > 9 else str(category)
                if column_name + '_' + cat not in df_dummies.columns:
                    df_dummies[column_name + '_' + cat] = False

            # Ordenar las columnes
            df_dummies = df_dummies.reindex(sorted(df_dummies.columns), axis=1)

            if 0 not in all_categories and column_name + '_00' in df_dummies.columns:
                df_dummies.drop(columns=[column_name + '_00'], inplace=True)
            if 0 not in all_categories and column_name + '_0' in df_dummies.columns:
                df_dummies.drop(columns=[column_name + '_0'], inplace=True)

        # Concatenar les noves columnes després de la posició de la columna original
        original_column_position: int = df.columns.get_loc(column_name)
        df = pd.concat([df.iloc[:, :original_column_position], df_dummies, df.iloc[:, original_column_position + 1:]], axis=1)

        return df

    # Función para convertir un entero a una cadena binaria con 40 dígitos
    def __int_to_binary_string(self, num: int, total_inputs: int):
        return format(num, f'0{total_inputs}b')

    def __decimal_to_one_hot(self, df: DataFrame, column_name: str, total_inputs: int) -> DataFrame:
        # Aplicar la funció a la columna original i dividir la cadena binària en caràcters individuals
        df_binary = df[column_name].apply(lambda x: self.__int_to_binary_string(x, total_inputs)).apply(list).apply(pd.Series)

        # Canviar el nom a les noves columnes
        df_binary.columns = [column_name + "_" + str(i) for i in range(0, total_inputs)]

        # Concatenar les noves columnes després de la columna original
        original_column_position: int = df.columns.get_loc(column_name)
        df = pd.concat([df.iloc[:, :original_column_position], df_binary, df.iloc[:, original_column_position + 1:]], axis=1)

        return df

    def __convert_boolean_columns_to_integer(self, df: DataFrame) -> DataFrame:
        # Obtenir noms de columnes i tipus de dades
        boolean_columns_name = df.select_dtypes(include='bool').columns

        # Convertir columnas booleanas a enteros
        df[boolean_columns_name] = df[boolean_columns_name].astype(int)

        return df

    def __calc_intervals_points_combination(self, is_brisca: bool, num_players: int) -> list[int]:
        # Valors disponibles per a sumar
        values = [0, 2, 3, 4, 10, 11]

        # Generar combinacions possibles de sumands
        combinations = product(values, repeat=num_players)

        # Calcular les sumes úniques
        sums = set(sum(comb) for comb in combinations)

        # Calcular les sumes diferents en sumar 10 al resultat
        other_sums = set(one_sum + 10 for one_sum in sums)

        all_sums = sums.union(other_sums)

        if not is_brisca:
            other_sums_tute_1 = set(one_sum + 40 for one_sum in sums)
            other_sums_tute_2 = set(one_sum + 20 for one_sum in sums)
            other_sums_tute_3 = set(one_sum + 50 for one_sum in sums)

            all_sums = all_sums.union(other_sums_tute_1)
            all_sums = all_sums.union(other_sums_tute_2)
            all_sums = all_sums.union(other_sums_tute_3)

        # all_negative_sums = set(-one_sum for one_sum in all_sums)
        # all_sums = all_sums.union(all_negative_sums)

        # Imprimir las sumas únicas
        # print("Brisca, {} players:".format(num_players))
        # for points in sorted(all_sums):
        # print(sorted(all_sums))

        return sorted(all_sums)

    def __prepare_data(self) -> None:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)

        print("csv_filename load")
        df = pd.read_csv(self.__csv_filename)

        # Per algun motiu em mostra una columna al final, sense nom, plena de NaN
        df = df.iloc[:, :-1]

        if self.__csv_filename_2 != "":
            print("csv_filename_2 load")
            df2 = pd.read_csv(self.__csv_filename_2)

            # Per algun motiu em mostra una columna al final, sense nom, plena de NaN
            df2 = df2.iloc[:, :-1]

            df = pd.concat([df, df2], ignore_index=True)

        # Tractament de les dades
        # Intervals del nombre restant de cartes al deck
        # Deck size -> cada interval representarà una fase del joc
        # Brisca -> 2 jugadors: 0-34
        # 4 intervals (0)(0-6)(7-18)(19-34) // 0 = ultimes rondes, 0-6 = 3 ultimes rondes amb cartes restants
        # Brisca -> 3 jugadors: 0-31
        # 4 intervals (1)(2-10)(11-19)(20-31) // 1 = ultimes rondes, 2-10 = 3 ultimes rondes amb cartes restants
        # Brisca -> 4 jugadors: 0-28
        # 4 intervals (0)(0-8)(9-16)(17-28) // 0 = ultimes rondes, 0-8 = 2 ultimes rondes amb cartes restants

        # Tute -> 2 jugadors: 0-24
        # 3 intervals (0)(0-8)(9-16)(16-24)
        # Tute -> 3 jugadors: 0-16
        # 3 intervals (1)(2-7)(8-16) // 1 = ultimes rondes, 2-7 = 2 ultimes rondes amb cartes restants
        # Tute -> 4 jugadors: 0-8
        # 2 intervals (0)(0-8) // 0 = ultimes rondes, 0-8 = 2 ultimes rondes amb cartes restants

        #         if self.__is_brisca():
        #             if self.__num_players == 2:
        #                 intervals: List[int] = [-1, 0, 7, 19, 35]
        #                 labels: List[str] = ['deck_size_0', 'deck_size_1_6', 'deck_size_7_18', 'deck_size_19_34']
        #             elif self.__num_players == 3:
        #                 intervals = [0, 1, 11, 20, 32]
        #                 labels = ['deck_size_1', 'deck_size_2_10', 'deck_size_11_19', 'deck_size_20_31']
        #             else:
        #                 intervals = [-1, 0, 9, 17, 29]
        #                 labels = ['deck_size_0', 'deck_size_1_8', 'deck_size_9_16', 'deck_size_17_28']
        #         else:
        #             if self.__num_players == 2:
        #                 intervals = [-1, 0, 9, 17, 25]
        #                 labels = ['deck_size_0', 'deck_size_1_8', 'deck_size_9_16', 'deck_size_17_24']
        #             elif self.__num_players == 3:
        #                 intervals = [0, 1, 8, 17]
        #                 labels = ['deck_size_1', 'deck_size_2_7', 'deck_size_8_16']
        #             else:
        #                 intervals = [-1, 0, 9]
        #                 labels = ['deck_size_0', 'deck_size_1_8']
        #
        #         df = self.__dummy_vars(df, 'deck_size', intervals, labels, None)
        # Deck size no fa falta, ja sé les cartes que han sortit, afegir aquests inputs és redundant
        prepare_all_csv = True
        if prepare_all_csv:
            # Trump suit id -> 4 inputs
            # Trump label -> 10 inputs
            # One hot encoding
            all_categories_1_4 = [1, 2, 3, 4]
            all_categories_1_10 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            df = self.__dummy_vars(df, 'trump_suit_id', None, None, all_categories_1_4)
            df = self.__dummy_vars(df, 'trump_label', None, None, all_categories_1_10)

            # Cartes jugades previament
            # Suit id -> 4 inputs
            # Label -> 10 inputs
            for i in range(1, self.__num_players):
                df = self.__dummy_vars(df, f'played_card_{i}_suit_id', None, None, all_categories_1_4)
                df = self.__dummy_vars(df, f'played_card_{i}_label', None, None, all_categories_1_10)

            # Transformem inputs decimals a inputs binaris (One hot encoding)
            do_all_players: bool = False
            for i in range(0, self.__num_players):
                if do_all_players or i == 0:
                    # Ma del jugador
                    # Vector de 40 posicions fixes
                    # Transformar a binary i afegir 0 a l'esquerra + Crear vector one hot
                    df = self.__decimal_to_one_hot(df, f"hand_{i}_decimal_value", 40)
                else:
                    df = df.drop(f"hand_{i}_decimal_value", axis=1)

                # Cants dels jugadors
                # One hot encoding del suit id
                if not self.__is_brisca():
                    df = self.__dummy_vars(df, f"player_{i}_sing_suit_id", None, None, all_categories_1_4)

                # Puntuacio dels jugadors
                intervals = [-1, 20, 40, 60, 80, 100, 120, 300]
                labels = [f'player_{i}_score_0_19', f'player_{i}_score_20_39', f'player_{i}_score_40_59', f'player_{i}_score_60_79', f'player_{i}_score_80_99', f'player_{i}_score_100_119', f'player_{i}_score_120_max']
                df = self.__dummy_vars(df, f'player_{i}_score', intervals, labels, None)

            # Cartes vistes
            # Vector de 40 posicions fixes
            # Transformar a binary i afegir 0 a l'esquerra + Crear vector one hot
            df = self.__decimal_to_one_hot(df, "all_played_cards_decimal_value", 40)

            # Regles
            # Vector de 3 o 4 posicions fixes
            # Transformar a binary i afegir 0 a l'esquerra + Crear vector one hot
            total_rules = 4 if self.__num_players == 2 else 3
            df = self.__decimal_to_one_hot(df, "rules_decimal_value", total_rules)

            # Accions
            # Vector de 41 a 45 posicions fixes
            total_actions = 41 if self.__is_brisca() else 45
            # Transformar a binary i afegir 0 a l'esquerra + Crear vector one hot
            df = self.__decimal_to_one_hot(df, "actions_decimal_value", total_actions)

        # TODO -> haig de modificar això si vull fer tots els trainings alhoira
        # Aquesta primera part elimina "round_score", per tant no es pot utilitzar en l'heuristic

        heuristics = True
        if heuristics:
            # Normalitzar
            # max_value = df['heuristics'].max()
            # min_value = df['heuristics'].min()
            # df['heuristics_normalized'] = (df['heuristics'] - min_value) / (max_value - min_value)

            # Prova de combinar punts, win or lose i heuristocs
            # df['heuristics_normalized'] = df['round_score'] + (df['win_or_lose'] * 30) + df['heuristics']

            # Prova amb intervals per a la heursitica amb els punts
            # 10 intervals (Discretitzar)
            # num_intervals = 30
            # labels = ['heu_0', 'heu_1', 'heu_2', 'heu_3', 'heu_4', 'heu_5', 'heu_6', 'heu_7', 'heu_8', 'heu_9',
            #           'heu_10', 'heu_11', 'heu_12', 'heu_23', 'heu_14', 'heu_15', 'heu_16', 'heu_17', 'heu_18', 'heu_19'
            #           'heu_20', 'heu_21', 'heu_22', 'heu_23', 'heu_24', 'heu_25', 'heu_26', 'heu_27', 'heu_28', 'heu_29']
            # labels = [f"Intervalo {i + 1}: {intervals[i]:.2f} - {intervals[i + 1]:.2f}" for i in range(len(intervals) - 1)]
            # df['heuristics_points'] = df['round_score'] + df['heuristics']
            # df = self.__dummy_vars(df, 'heuristics_points', num_intervals, labels, None)

            # Calcular totes les possibles combinacions de punts i heuristiques per generar un input binari per cadascuna de les opcions
            df['heuristics_points'] = df['round_score'] + df['heuristics']
            # df['heuristics_points'] = df['round_score'] + df['heuristics'] + (20 * df['win_or_lose'])
            all_categories_points = self.__calc_intervals_points_combination(self.__is_brisca(), self.__num_players)
            print(all_categories_points)
            distinct_values = df['heuristics'].unique()
            all_categories_heuristics_points = []

            # Itera sobre els elements de la llista i els valors únics de la columna del DataFrame
            for points_value in all_categories_points:
                for heuristic_value in distinct_values:
                    # Calcula todas las combinaciones posibles de sumas y restas
                    all_categories_heuristics_points.append(points_value + heuristic_value)
                    # all_categories_heuristics_points.append(points_value - heuristic_value)

            max_possible_value = max(all_categories_heuristics_points)
            df['heuristics_points'] = df['heuristics_points'] + max_possible_value
            # Si no sumo el valor maxim possible, les columnes no queden ben ordenades (així elimino valors negatius)
            all_categories_heuristics_points = [x + max_possible_value for x in all_categories_heuristics_points]

            all_categories_heuristics_points.sort()
            all_categories_heuristics_points = list(set(all_categories_heuristics_points))

            print(sorted(df['heuristics_points'].unique()))
            print(all_categories_heuristics_points)

            df = self.__dummy_vars(df, 'heuristics_points', None, None, all_categories_heuristics_points)

        round_points = True
        if round_points:
            # Round score
            # Si s'utilitza com a etiqueta final, cada possible puntuació correspondrà a un input binari
            all_categories_points = self.__calc_intervals_points_combination(self.__is_brisca(), self.__num_players)
            # labels = list(map(str, intervals))

            # max_possible_value = max(all_categories_points)
            # df['round_score'] = df['round_score'] + max_possible_value
            # Si no sumo el valor maxim possible, les columnes no queden ben ordenades

            # all_categories_points = [x + max_possible_value for x in all_categories_points]

            df = self.__dummy_vars(df, 'round_score', None, None, all_categories_points)

        # Convertir valors de la columna a valors segons la puntuació
        # df['round_score_output'] = pd.cut(df['round_score'], bins=intervals, labels=labels, include_lowest=True)
        # df['round_score_output'] = pd.cut(df['round_score'], bins=intervals, include_lowest=True)
        # df.drop(columns=['round_score'], inplace=True)

        # Win or Lose
        # Aquesta etiqueta ja té els valors 0 i 1
        # Si s'utilitza com a etiqueta final no cal fer res

        # Es converteixen tots els camps booleans a enters (0, 1)
        df = self.__convert_boolean_columns_to_integer(df)

        # print(df)
        self.__df = df

        if self.__save_prepared_data:
            # Obtenir la extensió del arxiu
            extension = self.__csv_filename.split('.')[-1]

            # Afegir el sufix "save" abans de l'extensió
            if self.__csv_filename_2 != "":
                nom_arxiu = self.__csv_filename.replace('.' + extension, '_prepared_concat.' + extension)
            else:
                nom_arxiu = self.__csv_filename.replace('.' + extension, '_prepared.' + extension)

            # Guardar el DataFrame en format CSV
            df.to_csv(nom_arxiu, index=False, header=True)

    def __min_max_normalize(self, df: DataFrame, input_name: str, min_value: int, max_value: int) -> DataFrame:
        df[input_name] = (df[input_name] - min_value) / (max_value - min_value)
        return df

    def __prepare_data_normalized(self) -> None:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)

        print("csv_filename load")
        df = pd.read_csv(self.__csv_filename)

        # Per algun motiu em mostra una columna al final, sense nom, plena de NaN
        df = df.iloc[:, :-1]

        if self.__csv_filename_2 != "":
            print("csv_filename_2 load")
            df2 = pd.read_csv(self.__csv_filename_2)

            # Per algun motiu em mostra una columna al final, sense nom, plena de NaN
            df2 = df2.iloc[:, :-1]

            df = pd.concat([df, df2], ignore_index=True)

        # Tractament de les dades
        # Intervals del nombre restant de cartes al deck
        # Deck size -> cada interval representarà una fase del joc
        # Brisca -> 2 jugadors: 0-34
        # 4 intervals (0)(0-6)(7-18)(19-34) // 0 = ultimes rondes, 0-6 = 3 ultimes rondes amb cartes restants
        # Brisca -> 3 jugadors: 0-31
        # 4 intervals (1)(2-10)(11-19)(20-31) // 1 = ultimes rondes, 2-10 = 3 ultimes rondes amb cartes restants
        # Brisca -> 4 jugadors: 0-28
        # 4 intervals (0)(0-8)(9-16)(17-28) // 0 = ultimes rondes, 0-8 = 2 ultimes rondes amb cartes restants

        # Tute -> 2 jugadors: 0-24
        # 3 intervals (0)(0-8)(9-16)(16-24)
        # Tute -> 3 jugadors: 0-16
        # 3 intervals (1)(2-7)(8-16) // 1 = ultimes rondes, 2-7 = 2 ultimes rondes amb cartes restants
        # Tute -> 4 jugadors: 0-8
        # 2 intervals (0)(0-8) // 0 = ultimes rondes, 0-8 = 2 ultimes rondes amb cartes restants

        #         if self.__is_brisca():
        #             if self.__num_players == 2:
        #                 intervals: List[int] = [-1, 0, 7, 19, 35]
        #                 labels: List[str] = ['deck_size_0', 'deck_size_1_6', 'deck_size_7_18', 'deck_size_19_34']
        #             elif self.__num_players == 3:
        #                 intervals = [0, 1, 11, 20, 32]
        #                 labels = ['deck_size_1', 'deck_size_2_10', 'deck_size_11_19', 'deck_size_20_31']
        #             else:
        #                 intervals = [-1, 0, 9, 17, 29]
        #                 labels = ['deck_size_0', 'deck_size_1_8', 'deck_size_9_16', 'deck_size_17_28']
        #         else:
        #             if self.__num_players == 2:
        #                 intervals = [-1, 0, 9, 17, 25]
        #                 labels = ['deck_size_0', 'deck_size_1_8', 'deck_size_9_16', 'deck_size_17_24']
        #             elif self.__num_players == 3:
        #                 intervals = [0, 1, 8, 17]
        #                 labels = ['deck_size_1', 'deck_size_2_7', 'deck_size_8_16']
        #             else:
        #                 intervals = [-1, 0, 9]
        #                 labels = ['deck_size_0', 'deck_size_1_8']
        #
        #         df = self.__dummy_vars(df, 'deck_size', intervals, labels, None)
        # Deck size no fa falta, ja sé les cartes que han sortit, afegir aquests inputs és redundant
        prepare_all_csv = True
        if prepare_all_csv:
            # Trump suit id -> vector 4 posicions
            # Trump label -> valors 1 a 10
            # df = self.__min_max_normalize(df, 'trump_suit_id', 1, 4)
            all_categories_1_4 = [1, 2, 3, 4]
            df = self.__dummy_vars(df, 'trump_suit_id', None, None, all_categories_1_4)
            df = self.__min_max_normalize(df, 'trump_label', 1, 10)

            # Cartes jugades previament
            # Suit id -> vector 4 posicions
            # Label -> valors 0 a 10
            for i in range(1, self.__num_players):
                # df = self.__min_max_normalize(df, f'played_card_{i}_suit_id', 0, 4)
                df = self.__dummy_vars(df, f'played_card_{i}_suit_id', None, None, all_categories_1_4)
                df = self.__min_max_normalize(df, f'played_card_{i}_label', 0, 10)

            # Transformem inputs decimals a inputs binaris (One hot encoding)
            do_all_players: bool = False
            for i in range(0, self.__num_players):
                if do_all_players or i == 0:
                    # Ma del jugador
                    # Considerem entre 1 y 2^40 (valor hipotetic on tots els inputs estarien a 1, valor impossible perquè hi haurà entre 1 y 8 cartes a la mà)
                    df = self.__min_max_normalize(df, f"hand_{i}_decimal_value", 1, 2**40 - 1)
                else:
                    df = df.drop(f"hand_{i}_decimal_value", axis=1)

                # Cants dels jugadors -> valors 0 a 4
                if not self.__is_brisca():
                    df = self.__min_max_normalize(df, f"player_{i}_sing_suit_id", 0, 4)

                # Puntuacio dels jugadors (entre 0 y 130 brisca)
                # Puntuacio dels jugadors (entre 0 y 230 tute)
                max_score = 130 if self.__is_brisca() else 230
                df = self.__min_max_normalize(df, f'player_{i}_score', 0, max_score)

            # Cartes vistes
            # Considerem entre 0 y 2^40
            df = self.__min_max_normalize(df, "all_played_cards_decimal_value", 0, 2 ** 40 - 1)

            # Regles
            # Considerem entre 0 y 2^4 o 2^3
            total_rules = 2**4 - 1 if self.__num_players == 2 else 2**3 - 1
            df = self.__min_max_normalize(df, "rules_decimal_value", 0, total_rules)

            # Accions
            # Considerem entre 1 y 2^41 o 2^45
            # total_actions = 2**41 - 1 if self.__is_brisca() else 2**45 - 1
            # df = self.__min_max_normalize(df, "actions_decimal_value", 0, total_actions)

            # Vector de 41 a 45 posicions fixes
            total_actions = 41 if self.__is_brisca() else 45
            # Transformar a binary i afegir 0 a l'esquerra + Crear vector one hot
            df = self.__decimal_to_one_hot(df, "actions_decimal_value", total_actions)

        # TODO -> haig de modificar això si vull fer tots els trainings alhora
        # Aquesta primera part elimina "round_score", per tant no es pot utilitzar en l'heuristic

        heuristics = True
        if heuristics:
            # Normalitzar
            # max_value = df['heuristics'].max()
            # min_value = df['heuristics'].min()
            # df['heuristics_normalized'] = (df['heuristics'] - min_value) / (max_value - min_value)

            # Prova de combinar punts, win or lose i heuristocs
            # df['heuristics_normalized'] = df['round_score'] + (df['win_or_lose'] * 30) + df['heuristics']

            # Prova amb intervals per a la heursitica amb els punts
            # 10 intervals (Discretitzar)
            # num_intervals = 30
            # labels = ['heu_0', 'heu_1', 'heu_2', 'heu_3', 'heu_4', 'heu_5', 'heu_6', 'heu_7', 'heu_8', 'heu_9',
            #           'heu_10', 'heu_11', 'heu_12', 'heu_23', 'heu_14', 'heu_15', 'heu_16', 'heu_17', 'heu_18', 'heu_19'
            #           'heu_20', 'heu_21', 'heu_22', 'heu_23', 'heu_24', 'heu_25', 'heu_26', 'heu_27', 'heu_28', 'heu_29']
            # labels = [f"Intervalo {i + 1}: {intervals[i]:.2f} - {intervals[i + 1]:.2f}" for i in range(len(intervals) - 1)]
            # df['heuristics_points'] = df['round_score'] + df['heuristics']
            # df = self.__dummy_vars(df, 'heuristics_points', num_intervals, labels, None)

            # Calcular totes les possibles combinacions de punts i heuristiques per generar un input binari per cadascuna de les opcions
            df['heuristics_points'] = df['round_score'] + df['heuristics']
            # df['heuristics_points'] = df['round_score'] + df['heuristics'] + (20 * df['win_or_lose'])
            all_categories_points = self.__calc_intervals_points_combination(self.__is_brisca(), self.__num_players)
            print(all_categories_points)
            distinct_values = df['heuristics'].unique()
            all_categories_heuristics_points = []

            # Itera sobre els elements de la llista i els valors únics de la columna del DataFrame
            for points_value in all_categories_points:
                for heuristic_value in distinct_values:
                    # Calcula todas las combinaciones posibles de sumas y restas
                    all_categories_heuristics_points.append(points_value + heuristic_value)
                    # all_categories_heuristics_points.append(points_value - heuristic_value)

            max_possible_value = max(all_categories_heuristics_points)
            df['heuristics_points'] = df['heuristics_points'] + max_possible_value
            # Si no sumo el valor maxim possible, les columnes no queden ben ordenades (així elimino valors negatius)
            all_categories_heuristics_points = [x + max_possible_value for x in all_categories_heuristics_points]

            all_categories_heuristics_points.sort()
            all_categories_heuristics_points = list(set(all_categories_heuristics_points))

            print(sorted(df['heuristics_points'].unique()))
            print(all_categories_heuristics_points)

            df = self.__dummy_vars(df, 'heuristics_points', None, None, all_categories_heuristics_points)

        round_points = True
        if round_points:
            # Round score
            # Si s'utilitza com a etiqueta final, cada possible puntuació correspondrà a un input binari
            all_categories_points = self.__calc_intervals_points_combination(self.__is_brisca(), self.__num_players)
            # labels = list(map(str, intervals))

            # max_possible_value = max(all_categories_points)
            # df['round_score'] = df['round_score'] + max_possible_value
            # Si no sumo el valor maxim possible, les columnes no queden ben ordenades

            # all_categories_points = [x + max_possible_value for x in all_categories_points]

            df = self.__dummy_vars(df, 'round_score', None, None, all_categories_points)

        # Convertir valors de la columna a valors segons la puntuació
        # df['round_score_output'] = pd.cut(df['round_score'], bins=intervals, labels=labels, include_lowest=True)
        # df['round_score_output'] = pd.cut(df['round_score'], bins=intervals, include_lowest=True)
        # df.drop(columns=['round_score'], inplace=True)

        # Win or Lose
        # Aquesta etiqueta ja té els valors 0 i 1
        # Si s'utilitza com a etiqueta final no cal fer res

        # Es converteixen tots els camps booleans a enters (0, 1)
        df = self.__convert_boolean_columns_to_integer(df)

        # print(df)
        self.__df = df

        if self.__save_prepared_data:
            # Obtenir la extensió del arxiu
            extension = self.__csv_filename.split('.')[-1]

            # Afegir el sufix "save" abans de l'extensió
            if self.__csv_filename_2 != "":
                nom_arxiu = self.__csv_filename.replace('.' + extension, '_prepared_concat.' + extension)
            else:
                nom_arxiu = self.__csv_filename.replace('.' + extension, '_prepared_normalized.' + extension)

            # Guardar el DataFrame en format CSV
            df.to_csv(nom_arxiu, index=False, header=True)

    def __load_data(self) -> None:
        self.__df = pd.read_csv(self.__csv_filename)

    # TODO General -> Crec que puc eliminar el nombre de cartes restants al deck
    # Aquesta informació es solapa amb el total de cartes jugades
    # labels deck_size_X
    # TODO Round points -> En aquesta m'es indiferent la puntuació final, puc eliminar els inputs
    # labels = [f'player_{i}_score_0_19', f'player_{i}_score_20_39', f'player_{i}_score_40_59', f'player_{i}_score_60_79', f'player_{i}_score_80_99', f'player_{i}_score_100_119', f'player_{i}_score_120_max']
    def __training(self, training_type: int = None) -> None:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)

        training_type = self.__training_type if training_type is None else training_type

        # print("training_type", training_type)

        round_points: bool = True if training_type == 1 else False
        win_or_lose: bool = True if training_type == 2 else False
        heuristics: bool = True if training_type == 3 else False

        # labels supervisats
        if round_points:
            round_score_columns = [col for col in self.__df.columns if col.startswith('round_score')]
            # print(round_score_columns)
            dfy: DataFrame = self.__df[round_score_columns]
            # dfy: DataFrame = self.__df[["round_score"]]
            # dfy = self.__df[["win_or_lose"]]

            # dfy = np.asarray(dfy).astype(np.int)
            # dfy = np.array(dfy).astype(np.int_)

            #tf_data = {columna: tf.constant(df[columna].values, dtype=tf.int32) for columna in df.columns}
        elif heuristics:
            # dfy = self.__df[["heuristics_rp_wl"]]
            # dfy = self.__df[["heuristics_normalized"]]
            # heuristics_columns = [col for col in self.__df.columns if col.startswith('heu')]
            # print(round_score_columns)
            # dfy: DataFrame = self.__df[heuristics_columns]

            # Heuristic + round points -> 1 output per cada possible valor
            heuristic_round_score_columns = [col for col in self.__df.columns if col.startswith('heuristics_points')]
            # print(round_score_columns)
            # print(self.__df)
            dfy: DataFrame = self.__df[heuristic_round_score_columns]

            # print(dfy)
        else:
            dfy = self.__df[["win_or_lose"]]

        # Eliminem columnes restants
        # Agafem els noms de columnes que sí volem
        correct_columns = [col for col in self.__df.columns if not col.startswith('deck_size')]
        # Crea un nuevo DataFrame con solo las columnas que no tienen el sufijo especificado
        dfx = self.__df[correct_columns]

        correct_columns_y = [col for col in dfx.columns if not col.startswith('round_score')]
        dfx = dfx[correct_columns_y]

        correct_columns_y = [col for col in dfx.columns if not col.startswith('heuristics_points')]
        dfx = dfx[correct_columns_y]

        dfx = dfx.drop("win_or_lose", axis=1)
        dfx = dfx.drop("heuristics", axis=1)

        print(dfx)
        print(dfx.columns)

        dfx = np.asarray(dfx).astype(np.int_)

        # dfx = dfx.drop("round_score_output", axis=1)

        # print(dfx)
        # print(dfx.columns)

        # Dividim el conjunt de dades en entrenament i test (80 i 20%)
        x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.20)
        # print(x_train)
        # print(x_test)
        # Dividim el conjunt de test en test i validació (50 i 50%)
        x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, test_size=0.5)

        # Obtenir la extensió del arxiu
        date = datetime.now()
        nom_model = "sl_models/brisca/" if self.__is_brisca() else "sl_models/tute"
        # print("self.__only_assist", self.__only_assist)
        nom_model += "_only_assist/" if self.__only_assist else "/"
        nom_model += f"{self.__num_players}j/" if self.__single_mode else f"{self.__num_players}jt/"
        if round_points:
            nom_model = nom_model + "sl_rp_"
        elif heuristics:
            if 'normalized' in self.__csv_filename:
                nom_model = nom_model + "sl_heu_norm_"
            else:
                nom_model = nom_model + "sl_heu_"
        elif win_or_lose:
            nom_model = nom_model + "sl_wl_"

        nom_model += date.strftime("%Y%m%d_%H%M%S") + "_" + self.__save_filename + ".keras"

        # extension = self.__csv_filename.split('.')[-1]
        #         nom_model = self.__csv_filename
        #
        #         nom_model = nom_model.replace('data', 'sl_models')
        #         # print(nom_model)
        #
        #         if round_points:
        #             nom_model = nom_model.replace('jt/', 'jt/sl_rp_' + self.__save_filename)
        #             nom_model = nom_model.replace('j/', 'j/sl_rp_' + self.__save_filename)
        #         elif heuristics:
        #             nom_model = nom_model.replace('jt/', 'jt/sl_heu_' + self.__save_filename)
        #             nom_model = nom_model.replace('j/', 'j/sl_heu_' + self.__save_filename)
        #         else:
        #             nom_model = nom_model.replace('jt/', 'jt/sl_wl_' + self.__save_filename)
        #             nom_model = nom_model.replace('j/', 'j/sl_wl_' + self.__save_filename)
        #
        #         nom_model = nom_model.replace(extension, 'keras')
        #         nom_model = nom_model.replace('_save', '')
        print(nom_model)

        csv_logger = CustomCSVLogger(self.__csv_filename, nom_model, 'sl_models/training_log.csv', separator=',', append=True)

        # Definir l'arquitectura del model
        inputs: int = dfx.shape[1]

        layers_definition = []
        if self.layers is None:
            layers_definition = [
                layers.Dense(inputs // 2, activation='relu', input_shape=(inputs,)),
                layers.Dense(inputs // 4, activation='relu'),
                layers.Dense(inputs // 6, activation='relu'),
            ]
        else:
            for idx_layer, layer in enumerate(self.layers):
                if idx_layer == 0:
                    layers_definition.append(layers.Dense(layer, activation='relu', input_shape=(inputs,)))
                else:
                    layers_definition.append(layers.Dense(layer, activation='relu'))

        if win_or_lose:
            outputs = 1

            layers_definition.append(layers.Dense(outputs, activation='sigmoid'))  # Sortida binària

            model = models.Sequential(layers_definition)

            # S'haurà de tenir en compte quines mètriques són més eficients per a cada cas
            # F1 score i precision son ideals per a dades desbalancejades
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
            model.fit(x_train, y_train, epochs=10, validation_data=(x_validation, y_validation), callbacks=[csv_logger])
            # test_loss, test_accuracy, test_precision, test_recall, test_f1 = model.evaluate(x_test, y_test)
            test_loss, test_accuracy, test_precision, test_recall = model.evaluate(x_test, y_test)
        elif heuristics:
            # outputs = 1
            # outputs = 30
            outputs = y_train.shape[1]

            layers_definition.append(layers.Dense(outputs, activation='softmax'))   # Capa de sortida)

            model = models.Sequential(layers_definition)

            # S'haurà de tenir en compte quines mètriques són més eficients per a cada cas
            # F1 score i precision son ideals per a dades desbalancejades
            # model.compile(optimizer='adam', loss='mse', metrics=['accuracy', 'precision', 'recall', 'f1_score'])
            # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'precision', 'recall', 'f1_score'])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
            model.fit(x_train, y_train, epochs=10, validation_data=(x_validation, y_validation), callbacks=[csv_logger])
            # test_loss, test_accuracy, test_precision, test_recall, test_f1 = model.evaluate(x_test, y_test)
            test_loss, test_accuracy, test_precision, test_recall = model.evaluate(x_test, y_test)
        else:
            print("round points else")
            # outputs = 5 if self.__is_brisca() else 7
            # print(dfy)
            # print(y_train['round_score_output'].unique())
            # y_train_categorical = to_categorical(y_train, num_classes=outputs)
            #y_test_categorical = to_categorical(y_test, num_classes=outputs)
            # y_validation_categorical = to_categorical(y_validation, num_classes=outputs)

            # outputs = y_train_categorical.shape[1]
            outputs = y_train.shape[1]
            # print(outputs)
            # print(y_train)
            # print(outputs)

            layers_definition.append(layers.Dense(outputs, activation='softmax'))  # Capa de sortida)

            model = models.Sequential(layers_definition)

            # F1 score i precision son ideals per a dades desbalancejades
            # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'precision', 'recall', 'f1_score'])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
            # model.fit(x_train, y_train_categorical, epochs=10, validation_data=(x_validation, y_validation_categorical), callbacks=[csv_logger])
            model.fit(x_train, y_train, epochs=10, validation_data=(x_validation, y_validation), callbacks=[csv_logger])
            # test_loss, test_accuracy, test_precision, test_recall, test_f1 = model.evaluate(x_test, y_test_categorical)
            # test_loss, test_accuracy, test_precision, test_recall, test_f1 = model.evaluate(x_test, y_test)
            test_loss, test_accuracy, test_precision, test_recall = model.evaluate(x_test, y_test)

        print('Loss en el conjunto de prueba:', test_loss)
        print('Accuracy en el conjunto de prueba:', test_accuracy)
        print('Precision en el conjunto de prueba:', test_precision)
        print('Recall en el conjunto de prueba:', test_recall)
        # print('F1 en el conjunto de prueba:', test_f1)

        # Abrir el archivo CSV en modo de añadir ('a')
        with open('sl_models/training_log.csv', 'a', newline='') as file:
            writer = csv.writer(file)

            # Escribir las métricas en una nueva fila
            writer.writerow(
                # [self.__csv_filename, nom_model, 'Final', test_accuracy, test_loss, test_precision, test_recall, test_f1])
                [self.__csv_filename, nom_model, 'Final', test_accuracy, test_loss, test_precision, test_recall])

        # TODO -> Haig de guardar la informació de les mètriques per a cada sl_models

        # Emmagatzemar el model

        model.save(nom_model)

        # Obtenir els pesos i bias del model (per algoritmes genetics)
        # pesos_capas = modelo_cargado.get_weights()






