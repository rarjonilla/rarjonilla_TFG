import copy
import csv
import random
from copy import deepcopy
from typing import Tuple, List, Optional


class Genetic_training_crossover:
    """Classe de Funcions d'encreuament per Xarxa neuronal"""
    def __init__(self, crossover_ratio: float):
        # Probabilitat d'encreuament
        self.__crossover_ratio: float = crossover_ratio

    def update_crossover_ratio(self, crossover_ratio: float):
        self.__crossover_ratio = crossover_ratio

    def crossover(self, parent_1_vectors: Tuple[List[List[float]], List[List[float]]], parent_2_vectors: Tuple[List[List[float]], List[List[float]]]) -> Tuple[Tuple[List[List[float]], List[List[float]]], Tuple[List[List[float]], List[List[float]]], Optional[str]]:
        if random.random() <= self.__crossover_ratio:
            # LLista de funcions de crossover
            # crossovers = [self.__one_point_crossover, self.__multi_point_crossover, self.__uniform_crossover, self.__average_crossover_and_flat_crossover, self.__multivariate_crossover, self.__linear_recombination_crossover, self.__arithmetic_crossover]
            crossovers = [self.__one_point_crossover, self.__multi_point_crossover, self.__uniform_crossover, self.__average_crossover_and_flat_crossover, self.__multivariate_crossover, self.__arithmetic_crossover]

            # Selecció de la funció de crossover aleatòria
            crossover_func = random.choice(crossovers)

            # Execució de la funció d'encreuament
            c1, c2 = crossover_func(parent_1_vectors, parent_2_vectors)
            return c1, c2, crossover_func.__name__
        else:
            # Es retornen còpies dels pares
            return deepcopy(parent_1_vectors), deepcopy(parent_2_vectors), None

    # Funcions de crossover
    def __one_point_crossover(self, parent_1_vectors: Tuple[List[List[float]], List[List[float]]], parent_2_vectors: Tuple[List[List[float]], List[List[float]]]) -> Tuple[Tuple[List[List[float]], List[List[float]]], Tuple[List[List[float]], List[List[float]]]]:
        # Capes per realitzar el crossover (les no seleccionades es passarà la capa sencera al fill 1 les del pare 1 i al fill 2 el del pare 2)
        total_selected_layers = random.randint(1, len(parent_1_vectors[0]))
        available_layers_position = list(range(len(parent_1_vectors[0])))
        selected_layers_position = []

        child_1_weights: List[List[float]] = []
        child_1_bias: List[List[float]] = []
        child_2_weights: List[List[float]] = []
        child_2_bias: List[List[float]] = []

        for _ in range(total_selected_layers):
            random_layer_position = random.choice(available_layers_position)
            selected_layers_position.append(random_layer_position)
            available_layers_position.remove(random_layer_position)

        for idx in range(len(parent_1_vectors[0])):
            parent_1_weights_vector: List[float] = parent_1_vectors[0][idx]
            parent_1_bias_vector: List[float] = parent_1_vectors[1][idx]
            parent_2_weights_vector: List[float] = parent_2_vectors[0][idx]
            parent_2_bias_vector: List[float] = parent_2_vectors[1][idx]

            # print("parent_1_weights_vector", parent_1_weights_vector)
            # print("parent_1_bias_vector", parent_1_bias_vector)
            # print("parent_2_weights_vector", parent_2_weights_vector)
            # print("parent_2_bias_vector", parent_2_bias_vector)

            if idx not in selected_layers_position:
                child_1_weights.append(parent_1_weights_vector)
                child_1_bias.append(parent_1_bias_vector)
                child_2_weights.append(parent_2_weights_vector)
                child_2_bias.append(parent_2_bias_vector)
            else:
                # Generar pùnt de tall per als pesos i bias
                weight_slice_point: int = random.randint(1, len(parent_1_weights_vector) - 1)
                bias_slice_point: int = random.randint(1, len(parent_1_bias_vector) - 1)

                # print("weight_slice_point", weight_slice_point)
                # print("bias_slice_point", bias_slice_point)

                # Fer el crossover entre els vectors
                child_1_w: List[float] = parent_1_weights_vector[:weight_slice_point] + parent_2_weights_vector[weight_slice_point:]
                child_1_b: List[float] = parent_1_bias_vector[:bias_slice_point] + parent_2_bias_vector[bias_slice_point:]
                child_2_w: List[float] = parent_2_weights_vector[:weight_slice_point] + parent_1_weights_vector[weight_slice_point:]
                child_2_b: List[float] = parent_2_bias_vector[:bias_slice_point] + parent_1_bias_vector[bias_slice_point:]

                child_1_weights.append(child_1_w)
                child_1_bias.append(child_1_b)
                child_2_weights.append(child_2_w)
                child_2_bias.append(child_2_b)

                # print("child_1_weights", child_1_weights)
                # print("child_1_bias", child_1_bias)
                # print("child_2_weights", child_2_weights)
                # print("child_2_bias", child_2_bias)

        return (child_1_weights, child_1_bias), (child_2_weights, child_2_bias)

    def __multi_point_crossover(self, parent_1_vectors: Tuple[List[List[float]], List[List[float]]], parent_2_vectors: Tuple[List[List[float]], List[List[float]]]) -> Tuple[ Tuple[List[List[float]], List[List[float]]], Tuple[List[List[float]], List[List[float]]]]:
        # Capes per realitzar el crossover (les no seleccionades es passarà la capa sencera al fill 1 les del pare 1 i al fill 2 el del pare 2)
        total_selected_layers = random.randint(1, len(parent_1_vectors[0]))
        available_layers_position = list(range(len(parent_1_vectors[0])))
        selected_layers_position = []

        child_1_weights: List[List[float]] = []
        child_1_bias: List[List[float]] = []
        child_2_weights: List[List[float]] = []
        child_2_bias: List[List[float]] = []

        for _ in range(total_selected_layers):
            random_layer_position = random.choice(available_layers_position)
            selected_layers_position.append(random_layer_position)
            available_layers_position.remove(random_layer_position)

        for idx in range(len(parent_1_vectors[0])):
            parent_1_weights_vector: List[float] = parent_1_vectors[0][idx]
            parent_1_bias_vector: List[float] = parent_1_vectors[1][idx]
            parent_2_weights_vector: List[float] = parent_2_vectors[0][idx]
            parent_2_bias_vector: List[float] = parent_2_vectors[1][idx]

            # print("parent_1_weights_vector", parent_1_weights_vector)
            # print("parent_1_bias_vector", parent_1_bias_vector)
            # print("parent_2_weights_vector", parent_2_weights_vector)
            # print("parent_2_bias_vector", parent_2_bias_vector)

            if idx not in selected_layers_position:
                child_1_weights.append(parent_1_weights_vector)
                child_1_bias.append(parent_1_bias_vector)
                child_2_weights.append(parent_2_weights_vector)
                child_2_bias.append(parent_2_bias_vector)
            else:
                    # Obtenir un nombre aleatori de punts de tall entre 2 i la meitat del total de pesos del vector
                    # Nota: poso un limit mínim i màxim al nombre de talls (2 per no ser com one_point_crossover i la meitat del total de pesos)
                    total_weight_slice_points: int = random.randint(2, len(parent_1_weights_vector) // 2)
                    total_bias_slice_points: int = random.randint(2, len(parent_1_bias_vector) // 2)

                    # print("total_weight_slice_points", total_weight_slice_points)
                    # print("total_bias_slice_points", total_bias_slice_points)

                    # Generar els punts de tall únicos entre 0 i el total de pesos del vector
                    weight_slice_points: List[int] = sorted(random.sample(range(1, len(parent_1_weights_vector) - 1), total_weight_slice_points))
                    bias_slice_points: List[int] = sorted(random.sample(range(1, len(parent_1_bias_vector) - 1), total_bias_slice_points))

                    # S'afegeix el punt final
                    weight_slice_points.append(len(parent_1_weights_vector))
                    bias_slice_points.append(len(parent_1_bias_vector))

                    # print("weight_slice_points", weight_slice_points)
                    # print("bias_slice_points", bias_slice_points)

                    child_1_w: List[float] = []
                    child_1_b: List[float] = []
                    child_2_w: List[float] = []
                    child_2_b: List[float] = []

                    # Generar els intercanvis per als pesos
                    first_index: int = 0
                    idx = 0
                    for weight_slice_point in weight_slice_points:
                        if idx % 2 == 0:
                            child_1_w += parent_1_weights_vector[first_index:weight_slice_point]
                            child_2_w += parent_2_weights_vector[first_index:weight_slice_point]
                        else:
                            child_1_w += parent_2_weights_vector[first_index:weight_slice_point]
                            child_2_w += parent_1_weights_vector[first_index:weight_slice_point]

                        first_index = weight_slice_point
                        idx += 1

                    first_index = 0
                    idx = 0
                    for bias_slice_point in bias_slice_points:
                        if idx % 2 == 0:
                            child_1_b += parent_1_bias_vector[first_index:bias_slice_point]
                            child_2_b += parent_2_bias_vector[first_index:bias_slice_point]
                        else:
                            child_1_b += parent_2_bias_vector[first_index:bias_slice_point]
                            child_2_b += parent_1_bias_vector[first_index:bias_slice_point]

                        first_index = bias_slice_point
                        idx += 1

                    child_1_weights.append(child_1_w)
                    child_1_bias.append(child_1_b)
                    child_2_weights.append(child_2_w)
                    child_2_bias.append(child_2_b)

        # print("child_1_weights", child_1_weights)
        # print("child_1_bias", child_1_bias)
        # print("child_2_weights", child_2_weights)
        # print("child_2_bias", child_2_bias)

        return (child_1_weights, child_1_bias), (child_2_weights, child_2_bias)

    def __uniform_crossover(self, parent_1_vectors: Tuple[List[List[float]], List[List[float]]], parent_2_vectors: Tuple[List[List[float]], List[List[float]]]) -> Tuple[Tuple[List[List[float]], List[List[float]]], Tuple[List[List[float]], List[List[float]]]]:
        # Capes per realitzar el crossover (les no seleccionades es passarà la capa sencera al fill 1 les del pare 1 i al fill 2 el del pare 2)
        total_selected_layers = random.randint(1, len(parent_1_vectors[0]))
        available_layers_position = list(range(len(parent_1_vectors[0])))
        selected_layers_position = []

        child_1_weights: List[List[float]] = []
        child_1_bias: List[List[float]] = []
        child_2_weights: List[List[float]] = []
        child_2_bias: List[List[float]] = []

        for _ in range(total_selected_layers):
            random_layer_position = random.choice(available_layers_position)
            selected_layers_position.append(random_layer_position)
            available_layers_position.remove(random_layer_position)

        for idx in range(len(parent_1_vectors[0])):
            parent_1_weights_vector: List[float] = parent_1_vectors[0][idx]
            parent_1_bias_vector: List[float] = parent_1_vectors[1][idx]
            parent_2_weights_vector: List[float] = parent_2_vectors[0][idx]
            parent_2_bias_vector: List[float] = parent_2_vectors[1][idx]

            # print("parent_1_weights_vector", parent_1_weights_vector)
            # print("parent_1_bias_vector", parent_1_bias_vector)
            # print("parent_2_weights_vector", parent_2_weights_vector)
            # print("parent_2_bias_vector", parent_2_bias_vector)

            if idx not in selected_layers_position:
                child_1_weights.append(parent_1_weights_vector)
                child_1_bias.append(parent_1_bias_vector)
                child_2_weights.append(parent_2_weights_vector)
                child_2_bias.append(parent_2_bias_vector)
            else:
                child_1_w: List[float] = []
                child_1_b: List[float] = []
                child_2_w: List[float] = []
                child_2_b: List[float] = []

                # Generar els intercanvis per als pesos
                for i in range(len(parent_1_weights_vector)):
                    if random.randint(0, 1) == 0:
                        child_1_w.append(parent_1_weights_vector[i])
                        child_2_w.append(parent_2_weights_vector[i])
                    else:
                        child_1_w.append(parent_2_weights_vector[i])
                        child_2_w.append(parent_1_weights_vector[i])

                # Generar els intercanvis per als pesos
                for i in range(len(parent_1_bias_vector)):
                    if random.randint(0, 1) == 0:
                        child_1_b.append(parent_1_bias_vector[i])
                        child_2_b.append(parent_2_bias_vector[i])
                    else:
                        child_1_b.append(parent_2_bias_vector[i])
                        child_2_b.append(parent_1_bias_vector[i])

                child_1_weights.append(child_1_w)
                child_1_bias.append(child_1_b)
                child_2_weights.append(child_2_w)
                child_2_bias.append(child_2_b)

        # print("child_1_weights", child_1_weights)
        # print("child_1_bias", child_1_bias)
        # print("child_2_weights", child_2_weights)
        # print("child_2_bias", child_2_bias)

        return (child_1_weights, child_1_bias), (child_2_weights, child_2_bias)

    # Aquests generen 1 cadascun, per tant, executo 2 crossover per obtenir els dos fills
    def __average_crossover_and_flat_crossover(self, parent_1_vectors: Tuple[List[List[float]], List[List[float]]], parent_2_vectors: Tuple[List[List[float]], List[List[float]]]) -> Tuple[Tuple[List[List[float]], List[List[float]]], Tuple[List[List[float]], List[List[float]]]]:
        child_1_weights, child_1_bias = self.__average_crossover(parent_1_vectors, parent_2_vectors)
        child_2_weights, child_2_bias = self.__flat_crossover(parent_1_vectors, parent_2_vectors)
        return (child_1_weights, child_1_bias), (child_2_weights, child_2_bias)

    def __average_crossover(self, parent_1_vectors: Tuple[List[List[float]], List[List[float]]], parent_2_vectors: Tuple[List[List[float]], List[List[float]]]) -> Tuple[List[List[float]], List[List[float]]]:
        child_weights: List[List[float]] = []
        child_bias: List[List[float]] = []

        for idx in range(len(parent_1_vectors[0])):
            parent_1_weights_vector: List[float] = parent_1_vectors[0][idx]
            parent_1_bias_vector: List[float] = parent_1_vectors[1][idx]
            parent_2_weights_vector: List[float] = parent_2_vectors[0][idx]
            parent_2_bias_vector: List[float] = parent_2_vectors[1][idx]

            # print("parent_1_weights_vector", parent_1_weights_vector)
            # print("parent_1_bias_vector", parent_1_bias_vector)
            # print("parent_2_weights_vector", parent_2_weights_vector)
            # print("parent_2_bias_vector", parent_2_bias_vector)

            child_w: List[float] = []
            child_b: List[float] = []

            # Generar la mitja de cada pes
            for i in range(len(parent_1_weights_vector)):
                child_w.append((parent_1_weights_vector[i] + parent_2_weights_vector[i]) / 2)

            # Generar la mitja de cada bias
            for i in range(len(parent_1_bias_vector)):
                child_b.append((parent_1_bias_vector[i] + parent_2_bias_vector[i]) / 2)

            child_weights.append(child_w)
            child_bias.append(child_b)

            # print("child_1_weights", child_1_weights)
            # print("child_1_bias", child_1_bias)
            # print("child_2_weights", child_2_weights)
            # print("child_2_bias", child_2_bias)

        return child_weights, child_bias

    def __flat_crossover(self, parent_1_vectors: Tuple[List[List[float]], List[List[float]]], parent_2_vectors: Tuple[List[List[float]], List[List[float]]]) -> Tuple[List[List[float]], List[List[float]]]:
        child_weights: List[List[float]] = []
        child_bias: List[List[float]] = []

        for idx in range(len(parent_1_vectors[0])):
            parent_1_weights_vector: List[float] = parent_1_vectors[0][idx]
            parent_1_bias_vector: List[float] = parent_1_vectors[1][idx]
            parent_2_weights_vector: List[float] = parent_2_vectors[0][idx]
            parent_2_bias_vector: List[float] = parent_2_vectors[1][idx]

            # print("parent_1_weights_vector", parent_1_weights_vector)
            # print("parent_1_bias_vector", parent_1_bias_vector)
            # print("parent_2_weights_vector", parent_2_weights_vector)
            # print("parent_2_bias_vector", parent_2_bias_vector)

            child_w: List[float] = []
            child_b: List[float] = []

            # Generar la mitja de cada pes
            for weight_1, weight_2 in zip(parent_1_weights_vector, parent_2_weights_vector):
                child_w.append(random.uniform(weight_1, weight_2))

            for bias_1, bias_2 in zip(parent_1_bias_vector, parent_2_bias_vector):
                child_b.append(random.uniform(bias_1, bias_2))

            child_weights.append(child_w)
            child_bias.append(child_b)

        # print("child_1_weights", child_1_weights)
        # print("child_1_bias", child_1_bias)
        # print("child_2_weights", child_2_weights)
        # print("child_2_bias", child_2_bias)

        return child_weights, child_bias

    def __multivariate_crossover(self, parent_1_vectors: Tuple[List[List[float]], List[List[float]]], parent_2_vectors: Tuple[List[List[float]], List[List[float]]]) -> Tuple[Tuple[List[List[float]], List[List[float]]], Tuple[List[List[float]], List[List[float]]]]:
        # Capes per realitzar el crossover (les no seleccionades es passarà la capa sencera al fill 1 les del pare 1 i al fill 2 el del pare 2)
        total_selected_layers = random.randint(1, len(parent_1_vectors[0]))
        available_layers_position = list(range(len(parent_1_vectors[0])))
        selected_layers_position = []

        child_1_weights: List[List[float]] = []
        child_1_bias: List[List[float]] = []
        child_2_weights: List[List[float]] = []
        child_2_bias: List[List[float]] = []

        for _ in range(total_selected_layers):
            random_layer_position = random.choice(available_layers_position)
            selected_layers_position.append(random_layer_position)
            available_layers_position.remove(random_layer_position)

        for idx in range(len(parent_1_vectors[0])):
            parent_1_weights_vector: List[float] = parent_1_vectors[0][idx]
            parent_1_bias_vector: List[float] = parent_1_vectors[1][idx]
            parent_2_weights_vector: List[float] = parent_2_vectors[0][idx]
            parent_2_bias_vector: List[float] = parent_2_vectors[1][idx]

            # print("parent_1_weights_vector", parent_1_weights_vector)
            # print("parent_1_bias_vector", parent_1_bias_vector)
            # print("parent_2_weights_vector", parent_2_weights_vector)
            # print("parent_2_bias_vector", parent_2_bias_vector)

            if idx not in selected_layers_position:
                child_1_weights.append(parent_1_weights_vector)
                child_1_bias.append(parent_1_bias_vector)
                child_2_weights.append(parent_2_weights_vector)
                child_2_bias.append(parent_2_bias_vector)
            else:
                # Obtenir un nombre aleatori de punts de tall entre 2 i la meitat del total de pesos del vector
                # Nota: poso un limit mínim i màxim al nombre de talls (2 per no ser com one_point_crossover i la meitat del total de pesos)
                total_weight_slice_points: int = random.randint(2, len(parent_1_weights_vector) // 2)
                total_bias_slice_points: int = random.randint(2, len(parent_1_bias_vector) // 2)

                # print("total_weight_slice_points", total_weight_slice_points)
                # print("total_bias_slice_points", total_bias_slice_points)

                # Generar els punts de tall únicos entre 0 i el total de pesos del vector
                weight_slice_points: List[int] = sorted(random.sample(range(1, len(parent_1_weights_vector) - 1), total_weight_slice_points))
                bias_slice_points: List[int] = sorted(random.sample(range(1, len(parent_1_bias_vector) - 1), total_bias_slice_points))

                # S'afegeix el punt final
                weight_slice_points.append(len(parent_1_weights_vector))
                bias_slice_points.append(len(parent_1_bias_vector))

                # print("weight_slice_points", weight_slice_points)
                # print("bias_slice_points", bias_slice_points)

                child_1_w: List[float] = []
                child_1_b: List[float] = []
                child_2_w: List[float] = []
                child_2_b: List[float] = []

                # Generar els intercanvis per als pesos
                first_index: int = 0
                for weight_slice_point in weight_slice_points:
                    # print(first_index, weight_slice_point)
                    # Si un valor random és més petit que el crossover ratio, s'intercanvia aquest fragment del vector, sinó, es manté el fragment
                    if random.random() <= self.__crossover_ratio:
                        child_1_w += parent_2_weights_vector[first_index:weight_slice_point]
                        child_2_w += parent_1_weights_vector[first_index:weight_slice_point]
                    else:
                        child_1_w += parent_1_weights_vector[first_index:weight_slice_point]
                        child_2_w += parent_2_weights_vector[first_index:weight_slice_point]

                    first_index = weight_slice_point

                first_index = 0
                for bias_slice_point in bias_slice_points:
                    if random.random() <= self.__crossover_ratio:
                        child_1_b += parent_2_bias_vector[first_index:bias_slice_point]
                        child_2_b += parent_1_bias_vector[first_index:bias_slice_point]
                    else:
                        child_1_b += parent_1_bias_vector[first_index:bias_slice_point]
                        child_2_b += parent_2_bias_vector[first_index:bias_slice_point]

                    first_index = bias_slice_point

                child_1_weights.append(child_1_w)
                child_1_bias.append(child_1_b)
                child_2_weights.append(child_2_w)
                child_2_bias.append(child_2_b)

        # print("child_1_weights", child_1_weights)
        # print("child_1_bias", child_1_bias)
        # print("child_2_weights", child_2_weights)
        # print("child_2_bias", child_2_bias)

        return (child_1_weights, child_1_bias), (child_2_weights, child_2_bias)

    def __linear_recombination_crossover(self, parent_1_vectors: Tuple[List[List[float]], List[List[float]]], parent_2_vectors: Tuple[List[List[float]], List[List[float]]]) -> Tuple[Tuple[List[List[float]], List[List[float]]], Tuple[List[List[float]], List[List[float]]]]:
        # Capes per realitzar el crossover (les no seleccionades es passarà la capa sencera al fill 1 les del pare 1 i al fill 2 el del pare 2)
        total_selected_layers = random.randint(1, len(parent_1_vectors[0]))
        available_layers_position = list(range(len(parent_1_vectors[0])))
        selected_layers_position = []

        child_1_weights: List[List[float]] = []
        child_1_bias: List[List[float]] = []
        child_2_weights: List[List[float]] = []
        child_2_bias: List[List[float]] = []

        for _ in range(total_selected_layers):
            random_layer_position = random.choice(available_layers_position)
            selected_layers_position.append(random_layer_position)
            available_layers_position.remove(random_layer_position)

        for idx in range(len(parent_1_vectors[0])):
            parent_1_weights_vector: List[float] = parent_1_vectors[0][idx]
            parent_1_bias_vector: List[float] = parent_1_vectors[1][idx]
            parent_2_weights_vector: List[float] = parent_2_vectors[0][idx]
            parent_2_bias_vector: List[float] = parent_2_vectors[1][idx]

            # print("parent_1_weights_vector", parent_1_weights_vector)
            # print("parent_1_bias_vector", parent_1_bias_vector)
            # print("parent_2_weights_vector", parent_2_weights_vector)
            # print("parent_2_bias_vector", parent_2_bias_vector)

            if idx not in selected_layers_position:
                child_1_weights.append(parent_1_weights_vector)
                child_1_bias.append(parent_1_bias_vector)
                child_2_weights.append(parent_2_weights_vector)
                child_2_bias.append(parent_2_bias_vector)
            else:
                child_1_w: List[float] = []
                child_1_b: List[float] = []
                child_2_w: List[float] = []
                child_2_b: List[float] = []

                # En comptes de generar 3 fills i comprovar el seu fitness, farem un random de 1 a 3 i no farem el que surti del random
                skip: int = random.randint(1, 3)
                print("skip", skip)
                if skip != 1:
                    print("skip != 1")
                    # First offspring: parent_one + parent_two
                    child_1_w = [x + y for x, y in zip(parent_1_weights_vector, parent_2_weights_vector)]
                    child_1_b = [x + y for x, y in zip(parent_1_bias_vector, parent_2_bias_vector)]
                if skip != 2:
                    # Second offspring: (1.5)*parent_one - (0.5)*parent_two
                    if len(child_1_weights) == 0:
                        print("skip != 2 and len == 0")
                        child_1_w = [(1.5 * x) - (0.5 * y) for x, y in zip(parent_1_weights_vector, parent_2_weights_vector)]
                        child_1_b = [(1.5 * x) - (0.5 * y) for x, y in zip(parent_1_bias_vector, parent_2_bias_vector)]
                    else:
                        print("skip != 2 and len != 0")
                        child_2_w = [(1.5 * x) - (0.5 * y) for x, y in zip(parent_1_weights_vector, parent_2_weights_vector)]
                        child_2_b = [(1.5 * x) - (0.5 * y) for x, y in zip(parent_1_bias_vector, parent_2_bias_vector)]
                if skip != 3:
                    # Third offspring: (-0.5)*parent_one + (1.5)*parent_two
                    print("skip != 3")
                    child_2_w = [(1.5 * y) - (0.5 * x) for x, y in zip(parent_1_weights_vector, parent_2_weights_vector)]
                    child_2_b = [(1.5 * y) - (0.5 * x) for x, y in zip(parent_1_bias_vector, parent_2_bias_vector)]

                child_1_weights.append(child_1_w)
                child_1_bias.append(child_1_b)
                child_2_weights.append(child_2_w)
                child_2_bias.append(child_2_b)

        # print("child_1_weights", child_1_weights)
        # print("child_1_bias", child_1_bias)
        # print("child_2_weights", child_2_weights)
        # print("child_2_bias", child_2_bias)

        return (child_1_weights, child_1_bias), (child_2_weights, child_2_bias)

    def __arithmetic_crossover(self, parent_1_vectors: Tuple[List[List[float]], List[List[float]]], parent_2_vectors: Tuple[List[List[float]], List[List[float]]]) -> Tuple[Tuple[List[List[float]], List[List[float]]], Tuple[List[List[float]], List[List[float]]]]:
        child_1_weights: List[List[float]] = []
        child_1_bias: List[List[float]] = []
        child_2_weights: List[List[float]] = []
        child_2_bias: List[List[float]] = []

        for idx in range(len(parent_1_vectors[0])):
            parent_1_weights_vector: List[float] = parent_1_vectors[0][idx]
            parent_1_bias_vector: List[float] = parent_1_vectors[1][idx]
            parent_2_weights_vector: List[float] = parent_2_vectors[0][idx]
            parent_2_bias_vector: List[float] = parent_2_vectors[1][idx]

            # print("parent_1_weights_vector", parent_1_weights_vector)
            # print("parent_1_bias_vector", parent_1_bias_vector)
            # print("parent_2_weights_vector", parent_2_weights_vector)
            # print("parent_2_bias_vector", parent_2_bias_vector)

            # En comptes de generar 3 fills i comprovar el seu fitness, farem un random de 1 a 3 i no farem el que surti del random
            alpha: float = random.random()

            child_1_w: List[float] = [((1 - alpha) * x) + (alpha * y) for x, y in zip(parent_1_weights_vector, parent_2_weights_vector)]
            child_1_b: List[float] = [((1 - alpha) * x) + (alpha * y) for x, y in zip(parent_1_bias_vector, parent_2_bias_vector)]
            child_2_w: List[float] = [((1 - alpha) * y) + (alpha * x) for x, y in zip(parent_1_weights_vector, parent_2_weights_vector)]
            child_2_b: List[float] = [((1 - alpha) * y) + (alpha * x) for x, y in zip(parent_1_bias_vector, parent_2_bias_vector)]

            child_1_weights.append(child_1_w)
            child_1_bias.append(child_1_b)
            child_2_weights.append(child_2_w)
            child_2_bias.append(child_2_b)

            # print("child_1_weights", child_1_weights)
            # print("child_1_bias", child_1_bias)
            # print("child_2_weights", child_2_weights)
            # print("child_2_bias", child_2_bias)

        return (child_1_weights, child_1_bias), (child_2_weights, child_2_bias)
