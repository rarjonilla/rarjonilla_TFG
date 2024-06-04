import concurrent.futures
import csv
import gc
import json
import os
import pickle
import queue
import random
import re
import shutil

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from game_environment.game import Non_playable_game
from training.genetic.genetic_training_best_population import Genetic_training_best_population
from training.genetic.genetic_training_cm_directive import Genetic_training_cm_directive
from training.genetic.genetic_training_cm_log import Genetic_training_cm_log
from training.genetic.genetic_training_crossover import Genetic_training_crossover
from training.genetic.genetic_training_crossover_rl import Genetic_training_crossover_rl
from training.genetic.genetic_training_mutation import Genetic_training_mutation
from training.genetic.genetic_training_mutation_rl import Genetic_training_mutation_rl


class Genetic_training_rl:
    """Classe Supervised Training"""

    def __init__(self, game_type: int, total_games: int, num_players: int, single_mode: bool, rules: Dict, old_data: bool,
                 start_from_old_data: bool, save_filename: str, population_size: int,
                 best_n_population: int, generations: int, next_generation: int, cm_directive: int,
                 custom_crossover_ratio: float, custom_mutation_ratio: float, finalized_generation: bool, threads: int,
                 double_tournament: bool, sl_model: Optional[str], elit_selection: bool, eps: float, eps_decrease: float, gamma: float, only_one_agent: bool) -> None:
        self.__game_type: int = game_type
        self.__total_games: int = total_games
        self.__num_players: int = num_players
        self.__single_mode: bool = single_mode
        self.__rules = rules
        self.__population_size: int = population_size
        self.__best_n_population: int = best_n_population
        self.__generations: int = generations
        self.__generation: int = next_generation
        self.__finalized_generation: bool = finalized_generation
        self.__threads: int = threads
        self.__double_tournament: bool = double_tournament
        self.__sl_model: Optional[str] = sl_model
        self.__elite_selection: bool = elit_selection

        self.__only_one_agent: bool = only_one_agent
        self.__eps: float = eps
        self.__eps_decrease: float = eps_decrease
        self.__gamma: float = gamma

        self.__genetic_training_cm_directive: Genetic_training_cm_directive = Genetic_training_cm_directive(cm_directive, custom_crossover_ratio, custom_mutation_ratio, generations)
        self.__genetic_training_mutation: Genetic_training_mutation_rl = Genetic_training_mutation_rl(0)
        self.__genetic_training_crossover: Genetic_training_crossover_rl = Genetic_training_crossover_rl(0)

        self.__save_only_best_every: int = 100

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
        # self.__old_models_directory = self.__directory + "old_models/"
        if not self.__old_data:
            # os.makedirs(self.__old_models_directory)
            os.makedirs(self.__best_models_directory)

        if self.__generation == 0 and self.__finalized_generation:
            self.__generate_initial_population()

        self.info_filename = 'training_info.json'
        self.__data: Dict = {
            'game_type': self.__game_type,
            'total_games': self.__total_games,
            'num_players': self.__num_players,
            'single_mode': self.__single_mode,
            'rules': self.__rules,
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
            'elite_selection': self.__elite_selection,
            'only_one_agent': self.__only_one_agent,
            'eps': self.__eps,
            'eps_decrease': self.__eps_decrease,
            'gamma': self.__gamma
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
        self.__data['only_one_agent'] = self.__only_one_agent
        self.__data['eps'] = self.__eps
        self.__data['eps_decrease'] = self.__eps_decrease
        self.__data['gamma'] = self.__gamma

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

    def __copy_trained_population(self) -> None:
        # Agafar totes les poblacions de l'entrenament anterior
        # print(self.__old_data_directory)
        all_files: List[str] = os.listdir(self.__old_data_directory)
        old_models: List[str] = [file for file in all_files if file.startswith("ga_generation_")]

        for model in old_models:
            new_name = re.sub(r"ga_generation_\d+", "ga_generation_0", model)
            old_model = os.path.join(self.__old_data_directory, model)
            new_model = os.path.join(self.__directory, new_name)
            shutil.copytree(old_model, new_model)

    def __generate_initial_population(self) -> None:
        if self.__start_from_old_data:
            # Copiar les xarxes neuronals de l'entrenament anterior i canviar-li el número de generació a 0
            self.__copy_trained_population()
        else:
            # Crear X carpetes
            for i in range(self.__population_size):
                model_name = f"ga_generation_0_nn_{i + 1}"
                # print(f"{self.__directory}{model_name}")
                os.makedirs(f"{self.__directory}{model_name}")

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

    def __get_population_q_pairs_policy(self, models_path: List[str]) -> List[Tuple[str, Dict, Dict, Dict, Dict]]:
        vectors: List[Tuple[Dict, Dict, Dict, Dict]] = []
        for model_path in models_path:
            with open(self.__directory + model_path + "/q.pkl", 'rb') as q_file:
                q: Dict = pickle.load(q_file)

            with open(self.__directory + model_path + "/pairs_visited.pkl", 'rb') as pairs_visites_file:
                pairs_visited: Dict = pickle.load(pairs_visites_file)

            with open(self.__directory + model_path + "/policy.pkl", 'rb') as policy_file:
                policy: Dict = pickle.load(policy_file)

            with open(self.__directory + model_path + "/actions.pkl", 'rb') as actions_file:
                actions: Dict = pickle.load(actions_file)

            vectors.append((model_path, q, pairs_visited, policy, actions))
        return vectors

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
        # TODO -> en comptes de pesos i biaixos haig d'agafar Q, policy i pairs_visited
        population_q_pairs_policy: Tuple[str, Dict, Dict, Dict, Dict] = self.__get_population_q_pairs_policy(models_path)

        initial_population_size: int = 0 if not self.__elite_selection else self.__best_n_population

        # Crear nova població fins arribar al nombre total
        for i in range(initial_population_size, self.__population_size, 2):
            # Seleccionar 2 individus qualsevols de la llista
            parent_1_params, parent_2_params = random.sample(population_q_pairs_policy, 2)

            # Aplicar el crossover i generar 2 fills en format vector
            child_1_params, child_2_params, crossover_func = self.__genetic_training_crossover.crossover((parent_1_params[1], parent_1_params[2], parent_1_params[3], parent_1_params[4]), (parent_2_params[1], parent_2_params[2], parent_2_params[3], parent_2_params[4]))

            # Aplicar el mutations als fills en format vector
            child_1_params, mutation_func_child_1 = self.__genetic_training_mutation.mutation(child_1_params)
            child_2_params, mutation_func_child_2 = self.__genetic_training_mutation.mutation(child_2_params)

            q_c1, pairs_visited_c1, policy_c1, actions_c1 = child_1_params
            q_c2, pairs_visited_c2, policy_c2, actions_c2 = child_2_params

            # Generar dos sl_models nous, de la mateixa arquitectura que la original
            child_1_model_name = f"ga_generation_{self.__generation}_nn_{i + 1}"
            os.makedirs(f"{self.__directory}{child_1_model_name}")

            with open(f"{self.__directory}{child_1_model_name}" + "/q.pkl", 'wb') as q_file:
                pickle.dump(q_c1, q_file)

            with open(f"{self.__directory}{child_1_model_name}" "/pairs_visited.pkl", 'wb') as pairs_visites_file:
                pickle.dump(pairs_visited_c1, pairs_visites_file)

            with open(f"{self.__directory}{child_1_model_name}" + "/policy.pkl", 'wb') as policy_file:
                pickle.dump(policy_c1, policy_file)

            with open(f"{self.__directory}{child_1_model_name}" + "/actions.pkl", 'wb') as actions_file:
                pickle.dump(actions_c1, actions_file)

            shutil.copy(self.__directory + parent_1_params[0] + "/info.pkl", f"{self.__directory}{child_1_model_name}")

            child_2_model_name = f"ga_generation_{self.__generation}_nn_{i + 2}"
            os.makedirs(f"{self.__directory}{child_2_model_name}")

            with open(f"{self.__directory}{child_2_model_name}" + "/q.pkl", 'wb') as q_file:
                pickle.dump(q_c2, q_file)

            with open(f"{self.__directory}{child_2_model_name}" "/pairs_visited.pkl", 'wb') as pairs_visites_file:
                pickle.dump(pairs_visited_c2, pairs_visites_file)

            with open(f"{self.__directory}{child_2_model_name}" + "/policy.pkl", 'wb') as policy_file:
                pickle.dump(policy_c2, policy_file)

            with open(f"{self.__directory}{child_2_model_name}" + "/actions.pkl", 'wb') as actions_file:
                pickle.dump(actions_c2, actions_file)

            shutil.copy(self.__directory + parent_2_params[0] + "/info.pkl", f"{self.__directory}{child_2_model_name}")

            # Afegir els crossovers i mutacions al log
            gt_cm_l_1: Genetic_training_cm_log = Genetic_training_cm_log(child_1_model_name, parent_1_params[0], parent_2_params[0], c_r, crossover_func, m_r, mutation_func_child_1)
            gt_cm_l_2: Genetic_training_cm_log = Genetic_training_cm_log(child_2_model_name, parent_1_params[0], parent_2_params[0], c_r, crossover_func, m_r, mutation_func_child_2)

            self.__write_log_cm(gt_cm_l_1.get_csv_line())
            self.__write_log_cm(gt_cm_l_2.get_csv_line())

    def __get_all_population(self) -> List[str]:
        # Agafar totes les poblacions
        all_files: List[str] = os.listdir(self.__directory)
        return [file for file in all_files if file.startswith("ga_generation_")]

    def __get_all_best_population(self) -> List[str]:
        # Agafar totes les poblacions
        all_files: List[str] = os.listdir(self.__best_models_directory)
        return [file for file in all_files if file.startswith("ga_generation_")]

    def __save_best_models(self, best_models: List[Dict]) -> None:
        # Llista de paths de les poblacions
        models_path: List[str] = [model["model_name"] for model in best_models]

        for model_path in models_path:
            if not self.__elite_selection:
                shutil.move(self.__directory + model_path, self.__best_models_directory + model_path)
            else:
                if os.path.exists(self.__best_models_directory + model_path):
                    shutil.rmtree(self.__best_models_directory + model_path)

                shutil.copytree(self.__directory + model_path, self.__best_models_directory + model_path)

    def __save_old_models(self, best_models: List[Dict]) -> None:
        all_files: List[str] = self.__get_all_population()
        models_path: List[str] = [model["model_name"] for model in best_models]

        for model_path in all_files:
            if model_path not in models_path:
                shutil.move(self.__directory + model_path, self.__old_models_directory)

    def __delete_old_models(self, best_models: List[Dict]) -> None:
        # all_files: List[str] = self.__get_all_best_population()
        all_files: List[str] = self.__get_all_population()

        models_path: List[str] = [model["model_name"] for model in best_models]

        for model_path in all_files:
            if model_path not in models_path:
                # shutil.move(self.__directory + model_path, self.__old_models_directory)
                # os.rmdir(self.__directory + model_path)
                shutil.rmtree(self.__directory + model_path)

    def __delete_best_models(self, best_models: List[Dict]) -> None:
        all_files: List[str] = self.__get_all_best_population()

        patron = re.compile(r"ga_generation_(\d+)_nn_(\d+)")

        for model_path in all_files:
            # Comprobar si el nombre del archivo coincide con el patrón
            coincidencia = patron.match(model_path)
            if coincidencia:
                # Extraer los valores de x y nn
                x = coincidencia.group(1)

                # Comprobar si x % 2 es diferente de 0
                if int(x) % self.__save_only_best_every != 0 or int(x) == 0:
                    # os.rmdir(self.__directory + model_path)
                    shutil.rmtree(self.__best_models_directory + model_path)

        # models_path: List[str] = [model["model_name"] for model in best_models]

        # for model_path in all_files:
            # if model_path not in models_path:
                # os.remove(self.__directory + model_path)

    def __simulate_games(self, best_population: Genetic_training_best_population) -> List[Dict]:
        rivals_model_type = [9, 9, 9, 9]

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
                # if idm_model == 0 or self.:
                rivals_model_name.append(self.__directory + model_name)
                # else:
                #    rivals_model_name.append(model_name)

            # executar la simulació de l'emparellament
            # print(self.__only_one_agent)
            game: Non_playable_game = Non_playable_game(self.__game_type, self.__total_games, rivals_model_type,
                                                        rivals_model_name, self.__num_players,
                                                        self.__single_mode,
                                                        self.__rules, True, None, self.__eps, self.__eps_decrease,
                                                        self.__gamma, self.__only_one_agent)
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
                # self.__save_old_models(best_models)
                self.__delete_old_models(final_best_models)

                # Guardar estat entrenament
                self.__finalized_generation = False
                self.__save_info_json()

            final_best_population: Genetic_training_best_population = Genetic_training_best_population(self.__best_n_population)
            final_best_models: List[Dict] = self.__simulate_games(final_best_population)

            # Moure els millors sl_models i la resta
            # self.__save_old_models(final_best_models)
            self.__delete_old_models(final_best_models)
            self.__save_best_models(final_best_models)

            # Crossover + mutacions
            self.__create_new_population(final_best_models)
            self.__delete_best_models(final_best_models)

            # Actualitzar fitxer info
            self.__data["generation"] = self.__generation

            # Guardar estat entrenament
            self.__finalized_generation = True
            self.__save_info_json()
