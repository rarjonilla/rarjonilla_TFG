import concurrent.futures
import csv
import gc
import json
import os
import queue
import random
import re
import shutil

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from keras.metrics import Precision, Recall, Accuracy
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

import tensorflow as tf

from game_environment.game import Non_playable_game
from training.genetic.genetic_training_best_population import Genetic_training_best_population
from training.genetic.genetic_training_cm_directive import Genetic_training_cm_directive
from training.genetic.genetic_training_cm_log import Genetic_training_cm_log
from training.genetic.genetic_training_crossover import Genetic_training_crossover
from training.genetic.genetic_training_mutation import Genetic_training_mutation


class Genetic_training:
    """Classe Supervised Training"""

    def __init__(self, game_type: int, total_games: int, num_players: int, single_mode: bool, rules: Dict, old_data: bool,
                 start_from_old_data: bool, save_filename: str, layers: List[int], population_size: int,
                 best_n_population: int, generations: int, next_generation: int, cm_directive: int,
                 custom_crossover_ratio: float, custom_mutation_ratio: float, finalized_generation: bool, threads: int,
                 double_tournament: bool, sl_model: Optional[str], elit_selection: bool) -> None:
        self.__game_type: int = game_type
        self.__total_games: int = total_games
        self.__num_players: int = num_players
        self.__single_mode: bool = single_mode
        self.__rules = rules
        self.__layers: List[int] = layers
        self.__population_size: int = population_size
        self.__best_n_population: int = best_n_population
        self.__generations: int = generations
        self.__generation: int = next_generation
        self.__finalized_generation: bool = finalized_generation
        self.__threads: int = threads
        self.__double_tournament: bool = double_tournament
        self.__sl_model: Optional[str] = sl_model
        self.__elite_selection: bool = elit_selection

        self.__genetic_training_cm_directive: Genetic_training_cm_directive = Genetic_training_cm_directive(cm_directive, custom_crossover_ratio, custom_mutation_ratio, generations)
        self.__genetic_training_mutation: Genetic_training_mutation = Genetic_training_mutation(0, -1, 1)
        self.__genetic_training_crossover: Genetic_training_crossover = Genetic_training_crossover(0)

        date = datetime.now()
        rules_str: str = ""
        rules_str += "1" if rules["can_change"] else "0"
        rules_str += "1" if rules["last_tens"] else "0"
        rules_str += "1" if rules["black_hand"] else "0"
        rules_str += "1" if rules["hunt_the_three"] and self.__num_players == 2 else "0"
        rules_str += "_"
        self.__save_filename: str = rules_str + date.strftime("%Y%m%d_%H%M%S") + "_" + save_filename if not old_data and not start_from_old_data else save_filename
        self.__old_filename: str = ""
        self.__old_data: bool = old_data
        self.__start_from_old_data: bool = start_from_old_data

        if start_from_old_data:
            self.__old_filename = self.__save_filename
            new_save_filename_split: List[str] = self.__save_filename.split("_")
            new_save_filename_split = new_save_filename_split[2:]
            self.__save_filename = date.strftime("%Y%m%d_%H%M%S") + "_" + "_".join(new_save_filename_split)

        self.__old_data_directory: str = ""
        self.__directory: str = self.__define_directory()

        self.__best_models_directory = self.__directory + "best_models/"
        self.__old_models_directory = self.__directory + "old_models/"
        if not self.__old_data:
            os.makedirs(self.__old_models_directory)
            os.makedirs(self.__best_models_directory)

        self.__inputs: int = self.__define_inputs()

        # Cada possible acció és un output
        # Brisca: 40 cartes + intercanvi de triomf
        # Tute: 40 cartes + intercanvi de triomf + 4 possibles cants
        self.__outputs = 41 if self.__is_brisca() else 45

        self.__layers_definition = self.__define_layers_definition()

        if self.__generation == 0 and self.__finalized_generation:
            self.__generate_initial_population()

        self.info_filename = 'training_info.json'
        self.__data: Dict = {
            'game_type': self.__game_type,
            'total_games': self.__total_games,
            'num_players': self.__num_players,
            'single_mode': self.__single_mode,
            'rules': self.__rules,
            'layers': self.__layers,
            'population_size': self.__population_size,
            'best_n_population': self.__best_n_population,
            'generations': self.__generations,
            'generation': self.__generation,
            'cm_directive': cm_directive,
            'custom_crossover_ratio': custom_crossover_ratio,
            'custom_mutation_ratio': custom_mutation_ratio,
            'save_filename': self.__save_filename,
            'finalized_generation': self.__finalized_generation,
            'double_tournament': self.__double_tournament,
            'threads': self.__threads,
            'sl_model': self.__sl_model,
            'elite_selection': self.__elite_selection
        }

        self.__game_log = 'game_log.csv'
        if not os.path.exists(self.__directory + self.__game_log):
            self.__write_game_log("Model name,Wins,Points")

        self.__log_cm_filename = 'log_crossover_mutation.csv'
        if not os.path.exists(self.__directory + self.__log_cm_filename):
            self.__write_log_cm("Model name,Parent model 1,Parent model 2,Crossover ratio,Crossover function,Mutation ratio,Mutation function")

        self.__save_info_json()

        self.__training()

    def __is_brisca(self) -> bool:
        return self.__game_type == 1

    # Funció per guardar un JSON amb la informació de l'entrenament
    def __save_info_json(self) -> None:
        self.__data["finalized_generation"] = self.__finalized_generation

        with open(self.__directory + self.info_filename, 'w') as f:
            json.dump(self.__data, f, indent=4)

    def __write_log_cm(self, csv_line) -> None:
        with open(self.__directory + self.__log_cm_filename, 'a', newline='') as csv_file:
            # Crear objecte writer
            csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_NONE, quotechar='')
            # nom fill, nom_pare_1, nom_pare_2, crossover_ratio, crossover_function, mutation_ratio (None), mutation_function (None)
            csv_writer.writerow(csv_line.split(','))

    def __write_game_log(self, csv_line) -> None:
        with open(self.__directory + self.__game_log, 'a', newline='') as csv_file:
            # Crear objecte writer
            csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_NONE, quotechar='')
            # nom model, wins, points
            csv_writer.writerow(csv_line.split(','))

    # # Funció per obrir un JSON amb la informació de l'entrenament
    #     def __load_info_json(self):
    #         with open(self.__directory + self.info_filename, 'r') as f:
    #             return json.load(f)

    def __define_directory(self) -> str:
        directory: str = "ga_models/"
        if self.__game_type == 1:
            directory += "brisca/"
        elif self.__only_assist:
            directory += "tute_only_assist/"
        else:
            directory += "tute/"

        directory += f"{self.__num_players}j"

        if not self.__single_mode:
            directory += "t"

        if self.__start_from_old_data:
            self.__old_data_directory = directory + "/" + self.__old_filename + "/"

        # Es crea un directori per cada entrenament, d'aquesta manera puc "continuar" un entrenament quan vulgui (no fer-ho tot seguit)
        directory += "/" + self.__save_filename + "/"

        if not self.__old_data:
            os.makedirs(directory)

        return directory

    def __define_inputs(self) -> int:
        inputs = 0
        if self.__is_brisca():
            if self.__num_players == 2:
                inputs = 112
            elif self.__num_players == 3:
                inputs = 146
            elif self.__num_players == 4:
                inputs = 167
        else:
            if self.__num_players == 2:
                inputs = 134
            elif self.__num_players == 3:
                inputs = 158
            elif self.__num_players == 4:
                inputs = 183

        return inputs

    def __define_layers_definition(self):
        # layers_definition = []
        #         for idx_layer, neurons in enumerate(self.__layers):
        #             if idx_layer == 0:
        #                 layers_definition.append({'neurons': neurons, 'activation': 'relu', 'input_shape': (self.__inputs,)})
        #                 layers_definition.append({'neurons': neurons, 'activation': 'relu', 'input_shape': (self.__inputs,)})
        #             else:
        #                 layers_definition.append({'neurons': neurons, 'activation': 'relu'})
        #
        #         layers_definition.append({'neurons': self.__outputs, 'activation': 'softmax'})

        layers_definition_v2 = []

        for idx_layer, neurons in enumerate(self.__layers):
            if idx_layer == 0:
                # Per defecte, keras utilitza Glorot uniform per als pesos i zeros per el bias
                # Es pot provar per defecte o afegint un bias random per que les pobalcions siguin encara més diferents entre elles
                # layers_definition_v2.append(layers.Dense(neurons, activation='relu', input_shape=(self.__inputs,)), kernel_initializer=tf.keras.initializers.glorot_normal(), bias_initializer=tf.keras.initializers.RandomNormal())
                layers_definition_v2.append(layers.Dense(neurons, activation='relu', input_shape=(self.__inputs,)))
            else:
                layers_definition_v2.append(layers.Dense(neurons, activation='relu'))

        layers_definition_v2.append(layers.Dense(self.__outputs, activation='softmax'))

        return layers_definition_v2

    def __generate_random_nn(self, name):
        model = tf.keras.Sequential(self.__layers_definition, name=name)
        #         for layer in self.__layers_definition:
        #             # add layer with random weight and bias
        #             # Per defecte, keras utilitza Glorot uniform per als pesos i zeros per el bias
        #             # Es pot provar per defecte o afegint un bias random per que les pobalcions siguin encara més diferents entre elles
        #             model.add(tf.keras.layers.Dense(units=layer['neurons'], activation=layer['activation'], kernel_initializer=tf.keras.initializers.glorot_normal(), bias_initializer=tf.keras.initializers.RandomNormal()))
        #             print(layer['neurons'], layer['activation'])
        #             # model.add(tf.keras.layers.Dense(units=layer['neurons'], activation=layer['activation']))

        # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'precision', 'recall', 'f1_score'])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'precision', 'recall'])

        return model

    def __copy_trained_population(self) -> None:
        # Agafar totes les poblacions de l'entrenament anterior
        # print(self.__old_data_directory)
        all_files: List[str] = os.listdir(self.__old_data_directory)
        old_models: List[str] = [file for file in all_files if file.endswith(".keras")]

        for model in old_models:
            new_name = re.sub(r"ga_generation_\d+", "ga_generation_0", model)
            old_model = os.path.join(self.__old_data_directory, model)
            new_model = os.path.join(self.__directory, new_name)
            shutil.copyfile(old_model, new_model)

    def __generate_initial_population(self) -> None:
        if self.__start_from_old_data:
            # Copiar les xarxes neuronals de l'entrenament anterior i canviar-li el número de generació a 0
            self.__copy_trained_population()
        else:
            # Crear X xarxes neuronals
            for i in range(self.__population_size):
                model_name = f"ga_generation_0_nn_{i + 1}"
                model = self.__generate_random_nn(model_name)
                # print(f"{self.__directory}{model_name}.keras")
                model.save(f"{self.__directory}{model_name}.keras")

    def __select_pairings(self, population: List[str]) -> List[List[str]]:
        random.shuffle(population)

        return [population[i:i + self.__num_players] for i in range(0, len(population), self.__num_players)]

    # # Obtenir pesos i bias de totes les capes del model en un unic vector
    #     def __get_weights_bias_as_vector(self, models_path: List[str]) -> List[Tuple[str, List[float], List[float]]]:
    #         vectors: List[Tuple[List[float], List[float]]] = []
    #         for model_path in models_path:
    #             model = load_model(self.__best_models_directory + model_path, custom_objects={'precision': Precision(), 'recall': Recall(), 'accuracy': Accuracy()})
    #
    #             weights_vector: List[float] = []
    #             bias_vector: List[float] = []
    #
    #             # Es recorren les capes del model
    #             for i, layer in enumerate(model.layers):
    #                 # S'agafa la matriu de connexions dels pesos i els bias
    #                 layer_weights, layer_bias = layer.get_weights()
    #
    #                 # Es recorre cada fila de la matriu de pesos
    #                 for j, row in enumerate(layer_weights):
    #                     # print("Capa", i + 1, j + 1, "Pesos:", row)
    #                     # S'afegeix cada pes al vector de pesos
    #                     for weight in row:
    #                         weights_vector.append(weight)
    #
    #                 # print("Capa", i + 1, "Bias:", layer_bias)
    #                 # S'afegeix cada bias al vector de pesos
    #                 for weight in layer_bias:
    #                     bias_vector.append(weight)
    #             vectors.append((model_path, weights_vector, bias_vector))
    #         return vectors

    # Obtenir pesos i bias de cada capa del model en un unic vector
    def __get_weights_bias_as_vector(self, models_path: List[str]) -> List[Tuple[str, List[List[float]], List[float]]]:
        vectors: List[Tuple[List[float], List[float]]] = []
        for model_path in models_path:
            model = load_model(self.__best_models_directory + model_path, custom_objects={'precision': Precision(), 'recall': Recall(), 'accuracy': Accuracy()})

            weights_vectors: List[List[float]] = []
            bias_vectors: List[List[float]] = []

            # Es recorren les capes del model
            for i, layer in enumerate(model.layers):
                weights_vector: List[float] = []
                bias_vector: List[float] = []

                # S'agafa la matriu de connexions dels pesos i els bias
                layer_weights, layer_bias = layer.get_weights()

                # Es recorre cada fila de la matriu de pesos
                for j, row in enumerate(layer_weights):
                    # print("Capa", i + 1, j + 1, "Pesos:", row)
                    # S'afegeix cada pes al vector de pesos
                    for weight in row:
                        weights_vector.append(weight)

                # print("Capa", i + 1, "Bias:", layer_bias)
                # S'afegeix cada bias al vector de pesos
                for weight in layer_bias:
                    bias_vector.append(weight)

                weights_vectors.append(weights_vector)
                bias_vectors.append(bias_vector)

            vectors.append((model_path, weights_vectors, bias_vectors))
        return vectors

    #     def __set_weights_bias_to_model(self, model, weights: List[float], bias: List[float]):
    #         # Definim el nombre de neurones a cada capa (inclosos inputs)
    #         architecture: List[int] = [self.__inputs] + self.__layers
    #         # print(architecture)
    #
    #         # Convertir vectors a np array
    #         weights_v = np.array(weights)
    #         bias_v = np.array(bias)
    #         # print(len(weights_v))
    #         # print(len(bias_v))
    #
    #         # Del vector, hem de generar la matriu de pesos i bias de cada capa
    #         # Seleccionarem els "n" elemenets de cada vector i en farem un reshape per aconseguir la matriu de la capa
    #         weight_matrix = []
    #         bias_vector = []
    #
    #         # Per tal poder seleccionar els n elements del vector, es guardarà les posicions que es van agafant
    #         weight_first_index = 0
    #         weight_last_index = 0
    #         bias_first_index = 0
    #         bias_last_index = 0
    #
    #         # Es recorren tots els nombres d'inputs i neurones de l'arquitectura de la xarxa neuronal excepte la sortida, que ja es té en compte en l'última iteració que es farà
    #         for idx_arc in range(len(architecture) - 1):
    #
    #             if idx_arc == 0:
    #                 # Per la primera iteració agafarem els "n" primers elements (el primer element de "architecture" és el nombre d'inputs)
    #                 # Calculem quants pesos corresponen a la primera capa (inputs * neurones de la capa)
    #                 weight_last_index = architecture[idx_arc] * architecture[idx_arc + 1]
    #                 # Calculem quants bias corresponen a la primera capa (neurones de la capa (les tenim a la següent posició de "architecture"))
    #                 bias_last_index = architecture[idx_arc + 1]
    #
    #                 # Afegim aquests elements a la matriu de pesos (fem el reshape amb el nombre d'inputs i el nombre de neurones)
    #                 # Per exemple, si tenim 10 inputs i 5 neurones, la matriu serà de la forma 10x5 i seleccionarem els primers 50 elements del vector
    #                 weight_matrix.append(weights_v[:weight_last_index].reshape(architecture[idx_arc], architecture[idx_arc + 1]))
    #                 # Fem el mateix per al bias, però com que ja és un vector, no cal fer el reshape
    #                 bias_vector.append(bias_v[:bias_last_index])
    #
    #                 # print("index", weight_first_index, weight_last_index)
    #                 # print("index_b", bias_first_index, bias_last_index)
    #                 # print("architecture", architecture[idx_arc], architecture[idx_arc + 1])
    #             else:
    #                 # Per a la resta d'iteracions agafarem els "n" següents elements (l'element actual de "architecture" conté el nombre de neurones de la capa anterior, i a la posició + 1 els de la capa actual)
    #                 # La posició index serà l'última posició d'elements de la iteració anterior
    #                 weight_first_index = weight_last_index
    #                 # Calculem el nombre de pesos de la capa i sumem la posició inicial
    #                 weight_last_index = architecture[idx_arc] * architecture[idx_arc + 1] + weight_first_index
    #                 # Fem el mateix amb la posició dels bias
    #                 bias_first_index = bias_last_index
    #                 bias_last_index = architecture[idx_arc + 1] + bias_first_index
    #
    #                 # Afegim aquests elements a la matriu de pesos (fem el reshape corresponent)
    #                 weight_matrix.append(weights_v[weight_first_index:weight_last_index].reshape(architecture[idx_arc], architecture[idx_arc + 1]))
    #                 # Fem el mateix per al bias, però com que ja és un vector, no cal fer el reshape
    #                 bias_vector.append(bias_v[bias_first_index:bias_last_index])
    #
    #                 # print("index", weight_first_index, weight_last_index)
    #                 # print("index_b", bias_first_index, bias_last_index)
    #                 # print("architecture", architecture[idx_arc], architecture[idx_arc + 1])
    #
    #         # print(weight_matrix)
    #         # print(bias_vector)
    #
    #         # Ara, es recorren les capes del model i es van associant amb les matrius de pesos i bias corresponents
    #         for layer, weights, bias in zip(model.layers, weight_matrix, bias_vector):
    #             layer.set_weights([weights, bias])
    #
    #         return model

    def __set_weights_bias_to_model(self, model, weights: List[List[float]], bias: List[List[float]]):
        # Definim el nombre de neurones a cada capa (inclosos inputs)
        architecture: List[int] = [self.__inputs] + self.__layers
        # print(architecture)

        # Convertir vectors a np array
        # weights_v = np.array(weights)
        # bias_v = np.array(bias)
        # print(len(weights_v))
        # print(len(bias_v))

        # Del vector, hem de generar la matriu de pesos i bias de cada capa
        # Seleccionarem els "n" elemenets de cada vector i en farem un reshape per aconseguir la matriu de la capa
        weight_matrix = []
        bias_vector = []

        # Per tal poder seleccionar els n elements del vector, es guardarà les posicions que es van agafant
        weight_first_index = 0
        weight_last_index = 0
        bias_first_index = 0
        bias_last_index = 0

        # Es recorren tots els nombres d'inputs i neurones de l'arquitectura de la xarxa neuronal excepte la sortida, que ja es té en compte en l'última iteració que es farà
        id_v = 0
        for idx_arc in range(len(architecture) - 1):
            # print("id_v", id_v)
            weights_v = np.array(weights[id_v])
            # print("len weights_v", len(weights_v))
            bias_v = np.array(bias[id_v])

            weight_matrix.append(weights_v.reshape(architecture[idx_arc], architecture[idx_arc + 1]))
            # Fem el mateix per al bias, però com que ja és un vector, no cal fer el reshape
            bias_vector.append(bias_v)

            id_v += 1

        # print(weight_matrix)
        # print(bias_vector)

        # Ara, es recorren les capes del model i es van associant amb les matrius de pesos i bias corresponents
        for layer, weights, bias in zip(model.layers, weight_matrix, bias_vector):
            layer.set_weights([weights, bias])

        return model

    def __create_new_population(self, best_models: List[Dict]) -> None:
        # Actualitzar generació
        self.__generation += 1

        # Actualitzar ratios
        c_r: float = self.__genetic_training_cm_directive.get_crossover_ratio(self.__generation)
        m_r: float = self.__genetic_training_cm_directive.get_mutation_ratio(self.__generation)
        self.__genetic_training_crossover.update_crossover_ratio(c_r)
        self.__genetic_training_mutation.update_mutation_ratio(m_r)

        # Recuperem la llista de pesos i bias de cada població seleccionada
        # Llista de paths de les poblacions
        models_path: List[str] = [model["model_name"] for model in best_models]
        # population_weight_and_bias_vectors: List[str, Tuple[List[float], List[float]]] = self.__get_weights_bias_as_vector(models_path)
        population_weight_and_bias_vectors: List[str, Tuple[List[List[float]], List[List[float]]]] = self.__get_weights_bias_as_vector(models_path)

        initial_population_size: int = 0 if not self.__elite_selection else self.__best_n_population

        # Crear nova població fins arribar al nombre total
        for i in range(initial_population_size, self.__population_size, 2):
            # Seleccionar 2 individus qualsevols de la llista
            parent_1_vectors, parent_2_vectors = random.sample(population_weight_and_bias_vectors, 2)

            # print("parent_1_vectors", parent_1_vectors)
            # print("parent_2_vectors", parent_2_vectors)

            # Aplicar el crossover i generar 2 fills en format vector
            child_1_vectors, child_2_vectors, crossover_func = self.__genetic_training_crossover.crossover((parent_1_vectors[1], parent_1_vectors[2]), (parent_2_vectors[1], parent_2_vectors[2]))

            # print("child_1_vectors", child_1_vectors)
            # print("child_2_vectors", child_2_vectors)

            # Aplicar el mutations als fills en format vector
            child_1_vectors, mutation_func_child_1 = self.__genetic_training_mutation.mutation(child_1_vectors)
            child_2_vectors, mutation_func_child_2 = self.__genetic_training_mutation.mutation(child_2_vectors)

            # print("child_1_vectors", child_1_vectors)
            # print("child_2_vectors", child_2_vectors)

            # Generar dos sl_models nous, de la mateixa arquitectura que la original
            child_1_model_name = f"ga_generation_{self.__generation}_nn_{i + 1}"
            child_2_model_name = f"ga_generation_{self.__generation}_nn_{i + 2}"
            child_1 = self.__generate_random_nn(child_1_model_name)
            child_2 = self.__generate_random_nn(child_2_model_name)

            # Afegir els vectors de pesos als sl_models
            # print(crossover_func, mutation_func_child_1, mutation_func_child_2)
            child_1_model = self.__set_weights_bias_to_model(child_1, child_1_vectors[0], child_1_vectors[1])
            child_2_model = self.__set_weights_bias_to_model(child_2, child_2_vectors[0], child_2_vectors[1])

            # Guardar els nous fills
            child_1_model.save(f"{self.__directory}{child_1_model_name}.keras")
            child_2_model.save(f"{self.__directory}{child_2_model_name}.keras")

            # Afegir els crossovers i mutacions al log
            gt_cm_l_1: Genetic_training_cm_log = Genetic_training_cm_log(child_1_model_name, parent_1_vectors[0], parent_2_vectors[0], c_r, crossover_func, m_r, mutation_func_child_1)
            gt_cm_l_2: Genetic_training_cm_log = Genetic_training_cm_log(child_2_model_name, parent_1_vectors[0], parent_2_vectors[0], c_r, crossover_func, m_r, mutation_func_child_2)

            self.__write_log_cm(gt_cm_l_1.get_csv_line())
            self.__write_log_cm(gt_cm_l_2.get_csv_line())

    def __get_all_population(self) -> List[str]:
        # Agafar totes les poblacions
        all_files: List[str] = os.listdir(self.__directory)
        return [file for file in all_files if file.endswith(".keras")]

    def __save_best_models(self, best_models: List[Dict]) -> None:
        # Llista de paths de les poblacions
        models_path: List[str] = [model["model_name"] for model in best_models]

        for model_path in models_path:
            if not self.__elite_selection:
                shutil.move(self.__directory + model_path, self.__best_models_directory)
            else:
                shutil.copy(self.__directory + model_path, self.__best_models_directory)

    def __save_old_models(self, best_models: List[Dict]) -> None:
        all_files: List[str] = self.__get_all_population()
        models_path: List[str] = [model["model_name"] for model in best_models]

        for model_path in all_files:
            if model_path not in models_path:
                shutil.move(self.__directory + model_path, self.__old_models_directory)

    def __simulate_games(self, best_population: Genetic_training_best_population) -> List[Dict]:
        rivals_model_type = [7, 7, 7, 7]
        if self.__sl_model is not None:
            rivals_model_type = [7, 8, 8, 8]

        # print(rivals_model_type)

        # Agafar tota la població
        population: List[str] = self.__get_all_population()

        pairings: List[List[str]] = []

        # Crear emparellaments random
        if self.__sl_model is None:
            # print("aaaaa")
            pairings = self.__select_pairings(population)
        else:
            # print("bbbbb")
            for idx, individual in enumerate(population):
                pairing: List[str] = []
                for i in range(self.__num_players):
                    if i != 0:
                        pairing.append(self.__sl_model)
                    else:
                        pairing.append(individual)
                # print("pairing", pairing)
                pairings.append(pairing)

        # Afegir-los a una cua
        pairings_queue = queue.Queue()
        for pairing in pairings:
            pairings_queue.put(pairing)

        # Funció per processar els emparellaments de la cua
        def process_pairing(pairing_) -> bool:
            # print(simulation_number)
            rivals_model_name: List[str] = []
            for idm_model, model_name in enumerate(pairing_):
                # print(model_name)
                if rivals_model_type[idm_model] == 7:
                    rivals_model_name.append(self.__directory + model_name)
                else:
                    rivals_model_name.append(model_name)

            # executar la simulació de l'emparellament
            game: Non_playable_game = Non_playable_game(self.__game_type, self.__total_games, rivals_model_type,
                                                        rivals_model_name, self.__num_players,
                                                        self.__single_mode,
                                                        self.__rules, False, None)
            # print("finalized", simulation_number)
            if self.__sl_model is None:
                for player_id in range(self.__num_players):
                    wins, points = game.get_player_wins_points(player_id)
                    best_population.add_model(pairing_[player_id], wins, points)
                    self.__write_game_log(f"{pairing_[player_id]}, {wins}, {points}")
            else:
                wins, points = game.get_player_wins_points(0)
                best_population.add_model(pairing_[0], wins, points)
                self.__write_game_log(f"{pairing_[0]}, {wins}, {points}")

            game.nullify_game()
            # game = None
            # del game
            # gc.collect()

            return True

        # Executar "n" simulacions alhora, quan acaba una simulació en començarà una de nova automàticament
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.__threads) as executor:
            futures = []

            while not pairings_queue.empty():
                # Llenar la lista de futuros con hasta 6 tareas de la cola
                while len(futures) < self.__threads and not pairings_queue.empty():
                    pairing = pairings_queue.get()
                    future = executor.submit(process_pairing, pairing)
                    futures.append(future)

                # Esperar a que al menos uno de los futuros termine
                concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)

                for future in concurrent.futures.as_completed(futures):
                    # print(future)
                    result: bool = future.result()
                    # print("future result", result)
                    futures.remove(future)

        return best_population.get_best_population()

    def __training(self) -> None:
        for i in range(self.__generation, self.__generations):
            # print("self.__directory", self.__directory)
            # print("population", population)
            if self.__finalized_generation and self.__double_tournament:
                best_population: Genetic_training_best_population = Genetic_training_best_population(self.__best_n_population * 2)
                best_models: List[Dict] = self.__simulate_games(best_population)

                # Moure els pitjors sl_models
                self.__save_old_models(best_models)

                # Guardar estat entrenament
                self.__finalized_generation = False
                self.__save_info_json()

            final_best_population: Genetic_training_best_population = Genetic_training_best_population(self.__best_n_population)
            final_best_models: List[Dict] = self.__simulate_games(final_best_population)

            # Moure els millors sl_models i la resta
            self.__save_old_models(final_best_models)
            self.__save_best_models(final_best_models)

            # Crossover + mutacions
            self.__create_new_population(final_best_models)

            # Actualitzar fitxer info
            self.__data["generation"] = self.__generation

            # Guardar estat entrenament
            self.__finalized_generation = True
            self.__save_info_json()
