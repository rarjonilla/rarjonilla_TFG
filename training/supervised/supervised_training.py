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

# TODO -> falta desenvolupar perquè l'usuari pugui triar les funcions d'activació i optimització que vulgui
class Supervised_training:
    """Classe Supervised Training"""

    def __init__(self, training_type: int, game_type: int, total_games: int, num_players: int, single_mode: bool, only_assist: bool, rivals_model_type: List[int], rivals_model_name: List[str], generate_data: bool, csv_filename: str, csv_filename_2: str, save_prepared_data: bool, save_filename: str, layers: List[int] = None, do_training: bool = True) -> None:
        # Tipus de sortida (1- Win or Lose, 2- Puntuacuó, 3- Puntuació + heurístic, 4- Totes les anteriors)
        self.__training_type: int = training_type

        # Informació de la modalitat
        self.__game_type: int = game_type
        self.__total_games: int = total_games
        self.__num_players: int = num_players
        self.__single_mode: bool = single_mode
        self.__only_assist = only_assist

        # Fitxer CSV on s'ha d'emmagatzemar el conjunt de dades (None) o que s'utilitzarà per a l'entrenament (not None)
        self.__csv_filename: str = "" if csv_filename is None else csv_filename
        # Nom d'un segon fitxer CSV per fer un merge de dos conjunts de dades diferents
        self.__csv_filename_2: str = "" if csv_filename_2 is None else csv_filename_2

        # Informació dels models per a la generació del conjunt de dades
        self.__rivals_model_type: List[int] = rivals_model_type
        self.__rivals_model_name: List[str] = rivals_model_name

        # Dataframe del cojunt de dades
        self.__df: Optional[DataFrame] = None

        # Nombre de neurones per capa
        self.layers: List[int] = layers

        # Indica si es vol emmagatzemar el conjunt de dades preparat (sense codificar)
        self.__save_prepared_data: bool = save_prepared_data
        # Nom amb el que es gaurdara el model / conjunt de dades
        self.__save_filename: str = save_filename

        # Indica si s'ha de preparar el conjunt de dades
        prepare_data: bool = False

        if generate_data:
            # Es genera el conjunt de dades
            print("generate")
            start_time_generate = time.time()
            self.__generate_data()
            print("--- %s seconds ---" % (time.time() - start_time_generate))

            prepare_data = True
        elif 'prepared' not in csv_filename:
            # S'indica que s'ha de preparar el conjunt de dades
            prepare_data = True

        if prepare_data:
            # Es prepara el conjunt de dades normalitzat
            print("prepare")
            start_time_prepare = time.time()
            # TODO - Ara només es fa normalitzant les entrades, s'hauria de demanar a l'usuari com vol fer-ho i afegir la condició aquí
            # self.__prepare_data()
            self.__prepare_data_normalized()
            print("--- %s seconds ---" % (time.time() - start_time_prepare))

        if csv_filename is not None and self.__df is None:
            # Es carrega el conjunt de dades
            print("load")
            start_time_prepare = time.time()
            self.__load_data()
            print("--- %s seconds ---" % (time.time() - start_time_prepare))

        if not prepare_data and csv_filename is None and self.__df is None:
            raise AttributeError("there is not data loaded")

        if do_training:
            # Es fa l'entrenament del model
            # if train:
            if training_type == 4:
                # Es genera un model per a cadascuna de les diferents sortides
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
                # Es genera un model per a l'entrenament corresponent
                print("train")
                start_time_train = time.time()
                self.__training()
                print("--- %s seconds ---" % (time.time() - start_time_train))

    def __is_brisca(self) -> bool:
        return self.__game_type == 1

    def __generate_data(self) -> None:
        # Generació del conjunt de dades. Es defineix el nom del fitxer CSV segons la modalitat
        date = datetime.now()
        if self.__csv_filename == "":
            self.__csv_filename = "data/brisca/" if self.__is_brisca() else "data/tute"
            self.__csv_filename += "_only_assist/" if self.__only_assist else "/"
            self.__csv_filename += f"{self.__num_players}j/" if self.__single_mode else f"{self.__num_players}jt/"
            self.__csv_filename += date.strftime("%Y%m%d_%H%M%S") + "_" + self.__save_filename + ".csv"

        # Es genera la capçalera del CSV
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

        # Es generen 8 o 16 combinacions de regles diferents
        # Cada CSV contindrà X partides per a cada combinació
        # Si decidim 1000 partides, es generaran 1000 partides per a cada combinació (8000 o 16000 en total)
        rules_str = "can_change last_tens black_hand"
        if self.__num_players == 2:
            rules_str += " hunt_the_three"

        rules_list = rules_str.split()

        for p in itertools.product([False, True], repeat=len(rules_list)):
            rules = dict(zip(rules_list, p))

            if self.__num_players > 2:
                rules['hunt_the_three'] = False

            rules['only_assist'] = self.__only_assist

            # Es genera el conjunt de dades per a cada combinació de regles
            game: Non_playable_game = Non_playable_game(self.__game_type, self.__total_games, self.__rivals_model_type, self.__rivals_model_name, self.__num_players, self.__single_mode, rules, True, self.__csv_filename, is_supervised_training=True)

    def __dummy_vars(self, df: DataFrame, column_name: str, intervals: Optional[List[int]], labels: Optional[List[str]], all_categories: Optional[List[int]]) -> DataFrame:
        # Funció que converteix els valors de les columnes especificades en un vector One Hot Encoding segons diferents paràmetres
        if intervals is not None and labels is not None:
            # Dividir la columna en intervals i crear variables binàries
            df_dummies = pd.get_dummies(pd.cut(df[column_name], bins=intervals, labels=labels))
        elif all_categories is not None:
            # Crear un input per a cada possible categoria
            if max(all_categories) > 9:
                df[column_name] = df[column_name].apply(lambda x: '{:02d}'.format(x))

            df_dummies = pd.get_dummies(df[column_name], prefix=column_name, columns=all_categories)

            # Afegir columnes restants, si és necessari (les que no estan presents en el dataframe)
            for category in all_categories:
                cat = '0'+str(category) if category < 10 and max(all_categories) > 9 else str(category)
                if column_name + '_' + cat not in df_dummies.columns:
                    df_dummies[column_name + '_' + cat] = False

            # Ordenar las columnes per nom
            df_dummies = df_dummies.reindex(sorted(df_dummies.columns), axis=1)

            # Eliminar les columnes si no han d'estar-hi
            if 0 not in all_categories and column_name + '_00' in df_dummies.columns:
                df_dummies.drop(columns=[column_name + '_00'], inplace=True)
            if 0 not in all_categories and column_name + '_0' in df_dummies.columns:
                df_dummies.drop(columns=[column_name + '_0'], inplace=True)

        # Concatenar les noves columnes després de la posició de la columna original
        original_column_position: int = df.columns.get_loc(column_name)
        df = pd.concat([df.iloc[:, :original_column_position], df_dummies, df.iloc[:, original_column_position + 1:]], axis=1)

        return df

    # Funció per a convertir un enter a una cadena binària amb X dígits
    def __int_to_binary_string(self, num: int, total_inputs: int):
        return format(num, f'0{total_inputs}b')

    def __decimal_to_one_hot(self, df: DataFrame, column_name: str, total_inputs: int) -> DataFrame:
        # Funció que transforma un valor decimal a codificació One Hot Encoding
        # Aplicar la funció a la columna original i dividir la cadena binària en caràcters individuals
        df_binary = df[column_name].apply(lambda x: self.__int_to_binary_string(x, total_inputs)).apply(list).apply(pd.Series)

        # Canviar el nom a les noves columnes
        df_binary.columns = [column_name + "_" + str(i) for i in range(0, total_inputs)]

        # Concatenar les noves columnes després de la columna original
        original_column_position: int = df.columns.get_loc(column_name)
        df = pd.concat([df.iloc[:, :original_column_position], df_binary, df.iloc[:, original_column_position + 1:]], axis=1)

        return df

    def __convert_boolean_columns_to_integer(self, df: DataFrame) -> DataFrame:
        # Es converteixen les columnes booleanes a valors enters (0- False, 1-True)
        # Obtenir noms de columnes i tipus de dades
        boolean_columns_name = df.select_dtypes(include='bool').columns

        # Convertir columnas booleanas a enteros
        df[boolean_columns_name] = df[boolean_columns_name].astype(int)

        return df

    def __calc_intervals_points_combination(self, is_brisca: bool, num_players: int) -> list[int]:
        # Funció que calcula totes les puntuacions diferents que es poden donar en una ronda segons el tipus de joc i el nombre de jugadors
        # Valors disponibles per a sumar
        values = [0, 2, 3, 4, 10, 11]

        # Generar combinacions possibles de sumands
        combinations = product(values, repeat=num_players)

        # Calcular les sumes úniques
        sums = set(sum(comb) for comb in combinations)

        # Calcular les sumes diferents en sumar 10 al resultat (10 d'últimes)
        other_sums = set(one_sum + 10 for one_sum in sums)

        all_sums = sums.union(other_sums)

        if not is_brisca:
            # S'afegeixen a les combinacions les sumes dels cants
            other_sums_tute_1 = set(one_sum + 40 for one_sum in sums)
            other_sums_tute_2 = set(one_sum + 20 for one_sum in sums)
            other_sums_tute_3 = set(one_sum + 50 for one_sum in sums)

            all_sums = all_sums.union(other_sums_tute_1)
            all_sums = all_sums.union(other_sums_tute_2)
            all_sums = all_sums.union(other_sums_tute_3)

        return sorted(all_sums)

    def __prepare_data(self) -> None:
        # Preparació del conjunt de dades no codificat a codificat (sense normalitzar)

        # Només per print del dataframe
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)

        print("csv_filename load")
        df = pd.read_csv(self.__csv_filename)

        # Per algun motiu, sempre em mostra una columna al final, sense nom i plena de NaN, s'elimina
        df = df.iloc[:, :-1]

        if self.__csv_filename_2 != "":
            print("csv_filename_2 load")
            df2 = pd.read_csv(self.__csv_filename_2)

            # Per algun motiu, sempre em mostra una columna al final, sense nom i plena de NaN, s'elimina
            df2 = df2.iloc[:, :-1]

            df = pd.concat([df, df2], ignore_index=True)

        # TRACTAMENT DE LES DADES

        # Intervals del nombre restant de cartes al deck
        # Deck size -> cada interval representarà una fase del joc
        # Deck size no fa falta, ja sé les cartes que han sortit per les cartes jugades al llarg de la partida, afegir aquests inputs és redundant

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
            # TODO - Si es vol tornar a activar les cartes conegudes del rival s'ha de posar a True
            # Amb False només fa la mà del jugador
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

        heuristics = True
        if heuristics:
            # Prova de normalitzar el valor heurístic (no ha funcionat)
            # max_value = df['heuristics'].max()
            # min_value = df['heuristics'].min()
            # df['heuristics_normalized'] = (df['heuristics'] - min_value) / (max_value - min_value)

            # Prova de combinar punts, win or lose i heuristocs (no ha funcionat)
            # df['heuristics_normalized'] = df['round_score'] + (df['win_or_lose'] * 30) + df['heuristics']

            # Prova amb intervals per a la heurística amb els punts (no ha funcionat)
            # 10 intervals (Discretitzar)
            # num_intervals = 30
            # labels = ['heu_0', 'heu_1', 'heu_2', 'heu_3', 'heu_4', 'heu_5', 'heu_6', 'heu_7', 'heu_8', 'heu_9',
            #           'heu_10', 'heu_11', 'heu_12', 'heu_23', 'heu_14', 'heu_15', 'heu_16', 'heu_17', 'heu_18', 'heu_19'
            #           'heu_20', 'heu_21', 'heu_22', 'heu_23', 'heu_24', 'heu_25', 'heu_26', 'heu_27', 'heu_28', 'heu_29']
            # labels = [f"Intervalo {i + 1}: {intervals[i]:.2f} - {intervals[i + 1]:.2f}" for i in range(len(intervals) - 1)]
            # df['heuristics_points'] = df['round_score'] + df['heuristics']
            # df = self.__dummy_vars(df, 'heuristics_points', num_intervals, labels, None)

            # Calcular totes les possibles combinacions de punts i heuristiques per generar un input binari per cadascuna de les opcions (sortida de la xarxa neuronal)
            df['heuristics_points'] = df['round_score'] + df['heuristics']
            # df['heuristics_points'] = df['round_score'] + df['heuristics'] + (20 * df['win_or_lose'])
            all_categories_points = self.__calc_intervals_points_combination(self.__is_brisca(), self.__num_players)
            # print(all_categories_points)
            distinct_values = df['heuristics'].unique()
            all_categories_heuristics_points = []

            # Itera sobre els elements de la llista i els valors únics de la columna del DataFrame
            for points_value in all_categories_points:
                for heuristic_value in distinct_values:
                    # Calcula totes les combinacions de sumes
                    all_categories_heuristics_points.append(points_value + heuristic_value)
                    # Calcula totes les combinacions de restes
                    # all_categories_heuristics_points.append(points_value - heuristic_value)

            max_possible_value = max(all_categories_heuristics_points)
            df['heuristics_points'] = df['heuristics_points'] + max_possible_value

            # Si no sumo el valor maxim possible, les columnes no queden ben ordenades (així elimino valors negatius)
            # -2, 0 5 -> 3, 5, 10
            all_categories_heuristics_points = [x + max_possible_value for x in all_categories_heuristics_points]

            all_categories_heuristics_points.sort()
            all_categories_heuristics_points = list(set(all_categories_heuristics_points))

            # print(sorted(df['heuristics_points'].unique()))
            # print(all_categories_heuristics_points)

            df = self.__dummy_vars(df, 'heuristics_points', None, None, all_categories_heuristics_points)

        round_points = True
        if round_points:
            # Round score
            # Si s'utilitza com a etiqueta final, cada possible puntuació correspondrà a un input binari
            all_categories_points = self.__calc_intervals_points_combination(self.__is_brisca(), self.__num_players)

            # S'ha tret la puntuació negativa, ja no cal això
            # max_possible_value = max(all_categories_points)
            # df['round_score'] = df['round_score'] + max_possible_value
            # Si no sumo el valor maxim possible, les columnes no queden ben ordenades
            # all_categories_points = [x + max_possible_value for x in all_categories_points]

            df = self.__dummy_vars(df, 'round_score', None, None, all_categories_points)

        # Win or Lose
        # Aquesta etiqueta ja té els valors 0 i 1
        # Si s'utilitza com a etiqueta final no cal fer res

        # Es converteixen tots els camps booleans a enters (0, 1)
        df = self.__convert_boolean_columns_to_integer(df)

        self.__df = df

        if self.__save_prepared_data:
            # Obtenir la extensió del arxiu
            extension = self.__csv_filename.split('.')[-1]

            # Afegir el sufix "prepared" abans de l'extensió
            if self.__csv_filename_2 != "":
                nom_arxiu = self.__csv_filename.replace('.' + extension, '_prepared_concat.' + extension)
            else:
                nom_arxiu = self.__csv_filename.replace('.' + extension, '_prepared.' + extension)

            # Guardar el DataFrame en format CSV
            df.to_csv(nom_arxiu, index=False, header=True)

    def __min_max_normalize(self, df: DataFrame, input_name: str, min_value: int, max_value: int) -> DataFrame:
        # Normalització per diferència entre màxim i mínim
        df[input_name] = (df[input_name] - min_value) / (max_value - min_value)
        return df

    def __prepare_data_normalized(self) -> None:
        # Preparació del conjunt de dades no codificat a codificat (amb valors normalitzats)

        # Només per print del dataframe
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)

        print("csv_filename load")
        df = pd.read_csv(self.__csv_filename)

        # Per algun motiu, sempre em mostra una columna al final, sense nom i plena de NaN, s'elimina
        df = df.iloc[:, :-1]

        if self.__csv_filename_2 != "":
            print("csv_filename_2 load")
            df2 = pd.read_csv(self.__csv_filename_2)

            # Per algun motiu em mostra una columna al final, sense nom, plena de NaN
            df2 = df2.iloc[:, :-1]

            df = pd.concat([df, df2], ignore_index=True)

        # Tractament de les dades
        # Intervals del nombre restant de cartes al deck
        # Deck size no fa falta, ja sé les cartes que han sortit per les cartes jugades al llarg de la partida, afegir aquests inputs és redundant

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

        prepare_all_csv = True
        if prepare_all_csv:
            # Trump suit id -> vector 4 posicions
            # Trump label -> valors 1 a 10 normalitzat
            all_categories_1_4 = [1, 2, 3, 4]
            df = self.__dummy_vars(df, 'trump_suit_id', None, None, all_categories_1_4)
            df = self.__min_max_normalize(df, 'trump_label', 1, 10)

            # Cartes jugades previament
            # Suit id -> vector 4 posicions
            # Label -> valors 0 a 10 normalitzat
            for i in range(1, self.__num_players):
                df = self.__dummy_vars(df, f'played_card_{i}_suit_id', None, None, all_categories_1_4)
                df = self.__min_max_normalize(df, f'played_card_{i}_label', 0, 10)

            # Transformem inputs decimals a normalitzats
            # TODO - Si es vol tornar a activar les cartes conegudes del rival s'ha de posar a True
            do_all_players: bool = False
            for i in range(0, self.__num_players):
                if do_all_players or i == 0:
                    # Ma del jugador
                    # Considerem entre 1 y 2^40 (valor hipotetic on tots els inputs estarien a 1, valor impossible perquè hi haurà entre 1 y 8 cartes a la mà)
                    df = self.__min_max_normalize(df, f"hand_{i}_decimal_value", 1, 2**40 - 1)
                else:
                    df = df.drop(f"hand_{i}_decimal_value", axis=1)

                # Cants dels jugadors -> valors 0 a 4 normalitzat
                if not self.__is_brisca():
                    df = self.__min_max_normalize(df, f"player_{i}_sing_suit_id", 0, 4)

                # Puntuacio dels jugadors (entre 0 y 130 brisca) normalitzat
                # Puntuacio dels jugadors (entre 0 y 230 tute) normalitzat
                max_score = 130 if self.__is_brisca() else 230
                df = self.__min_max_normalize(df, f'player_{i}_score', 0, max_score)

            # Cartes vistes
            # Considerem entre 0 y 2^40 normalitzat
            df = self.__min_max_normalize(df, "all_played_cards_decimal_value", 0, 2 ** 40 - 1)

            # Regles
            # Considerem entre 0 y 2^4 o 2^3 normalitzat
            total_rules = 2**4 - 1 if self.__num_players == 2 else 2**3 - 1
            df = self.__min_max_normalize(df, "rules_decimal_value", 0, total_rules)

            # Accions
            # Vector de 41 a 45 posicions fixes
            total_actions = 41 if self.__is_brisca() else 45

            # Transformar a binary i afegir 0 a l'esquerra + Crear vector one hot
            df = self.__decimal_to_one_hot(df, "actions_decimal_value", total_actions)

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
            # print(all_categories_points)
            distinct_values = df['heuristics'].unique()
            all_categories_heuristics_points = []

            # Itera sobre els elements de la llista i els valors únics de la columna del DataFrame
            for points_value in all_categories_points:
                for heuristic_value in distinct_values:
                    # Calcula totes les combinacions de sumes
                    all_categories_heuristics_points.append(points_value + heuristic_value)
                    # all_categories_heuristics_points.append(points_value - heuristic_value)

            max_possible_value = max(all_categories_heuristics_points)
            df['heuristics_points'] = df['heuristics_points'] + max_possible_value
            # Si no sumo el valor maxim possible, les columnes no queden ben ordenades (així elimino valors negatius)
            all_categories_heuristics_points = [x + max_possible_value for x in all_categories_heuristics_points]

            all_categories_heuristics_points.sort()
            all_categories_heuristics_points = list(set(all_categories_heuristics_points))

            # print(sorted(df['heuristics_points'].unique()))
            # print(all_categories_heuristics_points)

            df = self.__dummy_vars(df, 'heuristics_points', None, None, all_categories_heuristics_points)

        round_points = True
        if round_points:
            # Round score
            # Si s'utilitza com a etiqueta final, cada possible puntuació correspondrà a un input binari
            all_categories_points = self.__calc_intervals_points_combination(self.__is_brisca(), self.__num_players)

            # Amb puntuació només positiva això no cal
            # max_possible_value = max(all_categories_points)
            # df['round_score'] = df['round_score'] + max_possible_value
            # Si no sumo el valor maxim possible, les columnes no queden ben ordenades
            # all_categories_points = [x + max_possible_value for x in all_categories_points]

            df = self.__dummy_vars(df, 'round_score', None, None, all_categories_points)

        # Win or Lose
        # Aquesta etiqueta ja té els valors 0 i 1
        # Si s'utilitza com a etiqueta final no cal fer res

        # Es converteixen tots els camps booleans a enters (0, 1)
        df = self.__convert_boolean_columns_to_integer(df)

        self.__df = df

        if self.__save_prepared_data:
            # Obtenir la extensió del arxiu
            extension = self.__csv_filename.split('.')[-1]

            # Afegir el sufix "prepared" abans de l'extensió
            if self.__csv_filename_2 != "":
                nom_arxiu = self.__csv_filename.replace('.' + extension, '_prepared_concat.' + extension)
            else:
                nom_arxiu = self.__csv_filename.replace('.' + extension, '_prepared_normalized.' + extension)

            # Guardar el DataFrame en format CSV
            df.to_csv(nom_arxiu, index=False, header=True)

    def __load_data(self) -> None:
        self.__df = pd.read_csv(self.__csv_filename)

    def __training(self, training_type: int = None) -> None:
        # Només per print del dataframe
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)

        training_type = self.__training_type if training_type is None else training_type

        # Indiquem quin entrenament es farà
        round_points: bool = True if training_type == 1 else False
        win_or_lose: bool = True if training_type == 2 else False
        heuristics: bool = True if training_type == 3 else False

        # labels supervisats
        if round_points:
            # Posem al dataframe de les sortides les columnes de la puntuació
            round_score_columns = [col for col in self.__df.columns if col.startswith('round_score')]
            dfy: DataFrame = self.__df[round_score_columns]
        elif heuristics:
            # Posem al dataframe de les sortides les columnes de la puntuació + l'heurístic
            heuristic_round_score_columns = [col for col in self.__df.columns if col.startswith('heuristics_points')]
            dfy: DataFrame = self.__df[heuristic_round_score_columns]
        else:
            # Posem al dataframe de les sortides la columna amb win / lose
            dfy = self.__df[["win_or_lose"]]

        # Eliminem columnes restants per al dataframe de les entrades
        # Agafem els noms de columnes que sí volem
        correct_columns = [col for col in self.__df.columns if not col.startswith('deck_size')]
        dfx = self.__df[correct_columns]

        correct_columns_y = [col for col in dfx.columns if not col.startswith('round_score')]
        dfx = dfx[correct_columns_y]

        correct_columns_y = [col for col in dfx.columns if not col.startswith('heuristics_points')]
        dfx = dfx[correct_columns_y]

        dfx = dfx.drop("win_or_lose", axis=1)
        dfx = dfx.drop("heuristics", axis=1)

        # Transformem types de les columnes a enters
        dfx = np.asarray(dfx).astype(np.int_)

        # Dividim el conjunt de dades en entrenament i test (80 i 20%)
        x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.20)

        # Dividim el conjunt de test en test i validació (50 i 50%)
        x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, test_size=0.5)

        # Es defineix la ruta i nom del model
        date = datetime.now()
        nom_model = "sl_models/brisca/" if self.__is_brisca() else "sl_models/tute"
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

        # Es defineix el logger que emmagatzemarà el log de les mètriques per a cada epoch
        csv_logger = CustomCSVLogger(self.__csv_filename, nom_model, 'sl_models/training_log.csv', separator=',', append=True)

        # Definir l'arquitectura del model
        # TODO - Si es vol que l'usuari decideixi les funcions d'activacion, s'haurà de desenvolupar
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

        # Entrenament de Win / Lose
        if win_or_lose:
            # Només hi ha una sortida
            outputs = 1

            # Sortida binària
            layers_definition.append(layers.Dense(outputs, activation='sigmoid'))

            # Es crea el model
            model = models.Sequential(layers_definition)

            # Es compila el model
            # TODO - Si es vol que l'usuari decideixi la funció de loss i d'optimitzador, s'haurà de desenvolupar
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

            # S'entrena el model
            model.fit(x_train, y_train, epochs=10, validation_data=(x_validation, y_validation), callbacks=[csv_logger])

            # S'avalua el model
            test_loss, test_accuracy, test_precision, test_recall = model.evaluate(x_test, y_test)
        elif heuristics:
            # Tants outputs com combinacions possibles (la mida de columnes del conjunt de sortides)
            outputs = y_train.shape[1]

            # Sortida softmax per probabilitat de sortida (la suma de totes les sortides és 1)
            layers_definition.append(layers.Dense(outputs, activation='softmax'))

            # Es crea el model
            model = models.Sequential(layers_definition)

            # Es compila el model
            # TODO - Si es vol que l'usuari decideixi la funció de loss i d'optimitzador, s'haurà de desenvolupar
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

            # S'entrena el model
            model.fit(x_train, y_train, epochs=10, validation_data=(x_validation, y_validation), callbacks=[csv_logger])

            # S'avalua el model
            test_loss, test_accuracy, test_precision, test_recall = model.evaluate(x_test, y_test)
        else:
            # Tants outputs com puntuacuons possibles (la mida de columnes del conjunt de sortides)
            outputs = y_train.shape[1]

            # Sortida softmax per probabilitat de sortida (la suma de totes les sortides és 1)
            layers_definition.append(layers.Dense(outputs, activation='softmax'))  # Capa de sortida)

            # Es crea el model
            model = models.Sequential(layers_definition)

            # Es compila el model
            # TODO - Si es vol que l'usuari decideixi la funció de loss i d'optimitzador, s'haurà de desenvolupar
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

            # S'entrena el model
            model.fit(x_train, y_train, epochs=10, validation_data=(x_validation, y_validation), callbacks=[csv_logger])

            # S'avalua el model
            test_loss, test_accuracy, test_precision, test_recall = model.evaluate(x_test, y_test)

        # print('Loss en el conjunt de test:', test_loss)
        # print('Accuracy en el conjunt de test:', test_accuracy)
        # print('Precision en el conjunt de test:', test_precision)
        # print('Recall en el conjunt de test:', test_recall)

        # Afegir les mètriques de l'avaluació al CSV
        with open('sl_models/training_log.csv', 'a', newline='') as file:
            writer = csv.writer(file)

            # Escribir las métricas en una nueva fila
            writer.writerow(
                # [self.__csv_filename, nom_model, 'Final', test_accuracy, test_loss, test_precision, test_recall, test_f1])
                [self.__csv_filename, nom_model, 'Final', test_accuracy, test_loss, test_precision, test_recall])

        # Emmagatzemar el model
        model.save(nom_model)







