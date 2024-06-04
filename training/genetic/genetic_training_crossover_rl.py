import copy
import csv
import random
from copy import deepcopy
from typing import Tuple, List, Optional, Dict


class Genetic_training_crossover_rl:
    def __init__(self, crossover_ratio: float):
        self.__crossover_ratio: float = crossover_ratio

    def update_crossover_ratio(self, crossover_ratio: float):
        self.__crossover_ratio = crossover_ratio

    def crossover(self, parent_1_params: Tuple[Dict, Dict, Dict, Dict], parent_2_params: Tuple[Dict, Dict, Dict, Dict]) -> Tuple[Tuple[Dict, Dict, Dict, Dict], Tuple[Dict, Dict, Dict, Dict], str]:
        if random.random() <= self.__crossover_ratio:
            # LLista de funcions de crossover
            crossovers = [self.__one_point_crossover, self.__multi_point_crossover, self.__uniform_crossover, self.__multivariate_crossover]
            # crossovers = [self.__one_point_crossover, self.__multi_point_crossover]
            # crossovers = [self.__linear_recombination_crossover]

            q_p1, q_p2 = self.__merge_dicts(parent_1_params[0], parent_2_params[0])
            pairs_visited_p1, pairs_visited_p2 = self.__merge_dicts(parent_1_params[1], parent_2_params[1])
            policy_p1, policy_p2 = self.__merge_dicts(parent_1_params[2], parent_2_params[2])
            actions_p1, actions_p2 = self.__merge_dicts(parent_1_params[3], parent_2_params[3])

            # Selecció de la funció de crossover aleatòria
            crossover_func = random.choice(crossovers)
            c1, c2 = crossover_func((q_p1, pairs_visited_p1, policy_p1, actions_p1), (q_p2, pairs_visited_p2, policy_p2, actions_p2))
            return c1, c2, crossover_func.__name__
        else:
            # Es retornen còpies dels pares
            return deepcopy(parent_1_params), deepcopy(parent_2_params), None

    def __sort_dict(self, d: Dict):
        sorted_items = sorted(d.items())
        return dict(sorted_items)

    def __merge_dicts(self, dict_1: Dict, dict_2: Dict):
        d_1 = {}
        d_2 = {}

        for key, value in dict_1.items():
            if key not in dict_2:
                d_2[key] = value
            d_1[key] = value

        for key, value in dict_2.items():
            if key not in dict_1:
                d_1[key] = value
            d_2[key] = value

        d_1 = self.__sort_dict(d_1)
        d_2 = self.__sort_dict(d_2)

        return d_1, d_2

    def __one_point_crossover_for_dict(self, dict_1: Dict, dict_2: Dict, multiple_key: bool, crossover_point: int) -> Tuple[Dict, Dict]:
        d_1 = {}
        d_2 = {}

        if multiple_key:
            d1_p1 = {k: v for k, v in dict_1.items() if k[0] <= crossover_point}
            d1_p2 = {k: v for k, v in dict_1.items() if k[0] > crossover_point}
            d2_p1 = {k: v for k, v in dict_2.items() if k[0] <= crossover_point}
            d2_p2 = {k: v for k, v in dict_2.items() if k[0] > crossover_point}
        else:
            d1_p1 = {k: v for k, v in dict_1.items() if k <= crossover_point}
            d1_p2 = {k: v for k, v in dict_1.items() if v > crossover_point}
            d2_p1 = {k: v for k, v in dict_2.items() if k <= crossover_point}
            d2_p2 = {k: v for k, v in dict_2.items() if v > crossover_point}

        d_1.update(d1_p1)
        d_1.update(d2_p2)
        d_2.update(d2_p1)
        d_2.update(d1_p2)

        # No es eficient
        # for key, value in dict_1.items():

            # if (not multiple_key and key <= crossover_point) or (multiple_key and key[0] <= crossover_point):
                # d_1[key] = value
                # if key in dict_2:
                    # d_2[key] = dict_2[key]
        # else:
                # d_2[key] = value
                # if key in dict_2:
                    # d_1[key] = dict_2[key]

        return d_1, d_2

    # Funcions de crossover
    def __one_point_crossover(self, parent_1_params: Tuple[Dict, Dict, Dict, Dict], parent_2_params: Tuple[Dict, Dict, Dict, Dict]) -> Tuple[Tuple[Dict, Dict, Dict, Dict], Tuple[Dict, Dict, Dict, Dict]]:
        q_p1, pairs_visited_p1, policy_p1, actions_p1 = parent_1_params
        q_p2, pairs_visited_p2, policy_p2, actions_p2 = parent_2_params

        # Valors d'estat minim i màxim
        min_key = min(key for key in policy_p1.keys())
        max_key = max(key for key in policy_p1.keys())

        # Escollir un punt de tall per l'encreuament (s'escull un valor d'estat entre el minim i el maxim)
        crossover_point = random.randint(min_key, max_key - 1)
        # print(crossover_point)

        q_c1, q_c2 = self.__one_point_crossover_for_dict(q_p1, q_p2, True, crossover_point)
        pairs_visited_c1, pairs_visited_c2 = self.__one_point_crossover_for_dict(pairs_visited_p1, pairs_visited_p2, True, crossover_point)
        policy_c1, policy_c2 = self.__one_point_crossover_for_dict(policy_p1, policy_p2, False, crossover_point)

        return (q_c1, pairs_visited_c1, policy_c1, actions_p1), (q_c2, pairs_visited_c2, policy_c2, actions_p2)

    def __multi_point_crossover_for_dict(self, dict_1: Dict, dict_2: Dict, multiple_key: bool, crossover_points: List[int]) -> Tuple[Dict, Dict]:
        d_1 = {}
        d_2 = {}

        # Construir diccionaris alternanT entre intervals
        d1_p1 = {k: dict_1[k] for i, k in enumerate(dict_1) if any(crossover_points[j] <= i < crossover_points[j + 1] for j in range(len(crossover_points) - 1) if j % 2 == 0)}
        d1_p2 = {k: dict_1[k] for i, k in enumerate(dict_1) if any(crossover_points[j] <= i < crossover_points[j + 1] for j in range(len(crossover_points) - 1) if j % 2 != 0)}
        d2_p1 = {k: dict_2[k] for i, k in enumerate(dict_2) if any(crossover_points[j] <= i < crossover_points[j + 1] for j in range(len(crossover_points) - 1) if j % 2 == 0)}
        d2_p2 = {k: dict_2[k] for i, k in enumerate(dict_2) if any(crossover_points[j] <= i < crossover_points[j + 1] for j in range(len(crossover_points) - 1) if j % 2 != 0)}

        d_1.update(d1_p1)
        d_1.update(d2_p2)
        d_2.update(d2_p1)
        d_2.update(d1_p2)

        # No es eficient
        # Crear llistes per emmagatzemar els segments
        # d_1 = []
        # d_2 = []

        # Inicializar índex del punt de tall
        # crossover_index = 0

        # for key in sorted(dict_1.keys()):
            # Verificar si hem arribat a un punt de tall
            # if multiple_key:
                # if crossover_index < len(crossover_points) and key[0] >= crossover_points[crossover_index]:
                    # Alternar diccionaris
                    # dict_1, dict_2 = dict_2, dict_1
                    # Incrementar índex del punt de tall
                    # crossover_index += 1
            # else:
                # if crossover_index < len(crossover_points) and key >= crossover_points[crossover_index]:
                    # Alternar diccionaris
                    # dict_1, dict_2 = dict_2, dict_1
                    # Incrementar índex del punt de tall
                    # crossover_index += 1

            # Afegir claus a les llistes
            # if dict_1 is not None:
                # d_1.append((key, dict_1[key]))
            # if dict_2 is not None:
                # d_2.append((key, dict_2[key]))

        return dict(d_1), dict(d_2)

    def __multi_point_crossover(self, parent_1_params: Tuple[Dict, Dict, Dict, Dict], parent_2_params: Tuple[Dict, Dict, Dict, Dict]) -> Tuple[ Tuple[Dict, Dict, Dict, Dict], Tuple[Dict, Dict, Dict, Dict]]:
        q_p1, pairs_visited_p1, policy_p1, actions_p1 = parent_1_params
        q_p2, pairs_visited_p2, policy_p2, actions_p2 = parent_2_params

        # Punts d'encreuament
        num_points = random.randint(2, len(policy_p1.keys()) // 2)

        # Valors d'estat minim i màxim
        min_key = min(key for key in policy_p1.keys())
        max_key = max(key for key in policy_p1.keys())

        crossover_points = []
        for i in range(0, num_points):
            crossover_points.append(random.randint(min_key, max_key))

        crossover_points.sort()

        # print(crossover_points)

        q_c1, q_c2 = self.__multi_point_crossover_for_dict(q_p1, q_p2, True, crossover_points)
        pairs_visited_c1, pairs_visited_c2 = self.__multi_point_crossover_for_dict(pairs_visited_p1, pairs_visited_p2, True, crossover_points)
        policy_c1, policy_c2 = self.__multi_point_crossover_for_dict(policy_p1, policy_p2, False, crossover_points)

        return (q_c1, pairs_visited_c1, policy_c1, actions_p1), (q_c2, pairs_visited_c2, policy_c2, actions_p2)

    # No es eficient, s'elimina per poder fer proves
    def __uniform_crossover_for_dict(self, dict_1: Dict, dict_2: Dict) -> Tuple[Dict, Dict]:
        d_1 = {}
        d_2 = {}

        # Iterar sobre les keys i assignar aleatòriament
        for key in dict_1:
            if random.random() < 0.5:
                d_1[key] = dict_1[key]
                d_2[key] = dict_2[key]
            else:
                d_1[key] = dict_2[key]
                d_2[key] = dict_1[key]

        return d_1, d_2

    def __uniform_crossover(self, parent_1_params: Tuple[Dict, Dict, Dict, Dict], parent_2_params: Tuple[Dict, Dict, Dict, Dict]) -> Tuple[Tuple[Dict, Dict, Dict, Dict], Tuple[Dict, Dict, Dict, Dict]]:
        q_p1, pairs_visited_p1, policy_p1, actions_p1 = parent_1_params
        q_p2, pairs_visited_p2, policy_p2, actions_p2 = parent_2_params

        q_c1, q_c2 = self.__uniform_crossover_for_dict(q_p1, q_p2)
        pairs_visited_c1, pairs_visited_c2 = self.__uniform_crossover_for_dict(pairs_visited_p1, pairs_visited_p2)
        policy_c1, policy_c2 = self.__uniform_crossover_for_dict(policy_p1, policy_p2)

        return (q_c1, pairs_visited_c1, policy_c1, actions_p1), (q_c2, pairs_visited_c2, policy_c2, actions_p2)

    def __multivariate_crossover_for_dict(self, dict_1: Dict, dict_2: Dict, crossover_points: List[int]) -> Tuple[Dict, Dict]:
        d_1 = {}
        d_2 = {}

        keys_dict = list(dict_1.keys())

        # Iterar sobre segments entre els punts de tall
        start_index = 0
        for end_index in crossover_points:
            # Seleccionar pare pera al segment actual
            if random.random() < 0.5:
                parent = dict_1
            else:
                parent = dict_2

            # extreure segment del pare i asignarlo al fill corresponent
            for key in keys_dict[start_index:end_index]:
                if parent == dict_1:
                    d_1[key] = dict_1[key]
                    d_2[key] = dict_2[key]
                else:
                    d_1[key] = dict_2[key]
                    d_2[key] = dict_1[key]

            # Actualitzar índex d0inicio para el próximo segmento
            start_index = end_index

        # Asignar el último segmento
        for key in keys_dict[start_index:]:
            if parent == dict_1:
                d_1[key] = dict_1[key]
                d_2[key] = dict_2[key]
            else:
                d_1[key] = dict_2[key]
                d_2[key] = dict_1[key]

        return d_1, d_2

    # No es eficient, s'elimina per poder fer proves
    def __multivariate_crossover(self, parent_1_params: Tuple[Dict, Dict, Dict, Dict], parent_2_params: Tuple[Dict, Dict, Dict, Dict]) -> Tuple[Tuple[Dict, Dict, Dict, Dict], Tuple[Dict, Dict, Dict, Dict]]:
        q_p1, pairs_visited_p1, policy_p1, actions_p1 = parent_1_params
        q_p2, pairs_visited_p2, policy_p2, actions_p2 = parent_2_params

        keys_policy = policy_p1.keys()

        # Generar punts de tall
        crossover_points = random.sample(range(1, len(keys_policy)), random.randint(1, len(keys_policy) // 2))
        crossover_points.sort()
        # print(crossover_points)

        q_c1, q_c2 = self.__multivariate_crossover_for_dict(q_p1, q_p2, crossover_points)
        pairs_visited_c1, pairs_visited_c2 = self.__multivariate_crossover_for_dict(pairs_visited_p1, pairs_visited_p2, crossover_points)
        policy_c1, policy_c2 = self.__multivariate_crossover_for_dict(policy_p1, policy_p2, crossover_points)

        return (q_c1, pairs_visited_c1, policy_c1, actions_p1), (q_c2, pairs_visited_c2, policy_c2, actions_p2)
