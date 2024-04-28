import random
from typing import Tuple, List, Optional


class Genetic_training_mutation:
    def __init__(self, mutation_ratio: float, min_valid_value: int, max_valid_value: int):
        self.__mutation_ratio: float = mutation_ratio
        self.min_valid_value: int = min_valid_value
        self.max_valid_value: int = max_valid_value
        # Nota: poso un limit d'un 5% dels gens per a la mutació (és probable que aquest valor varii al fer proves)
        self.gens_mutation_ratio: float = 0.05

    def update_mutation_ratio(self, mutation_ratio: float):
        self.__mutation_ratio = mutation_ratio

    def mutation(self, child_vectors: Tuple[List[float], List[float]]) -> Tuple[Tuple[List[float], List[float]], Optional[str]]:
        if random.random() <= self.__mutation_ratio:
            # LLista de funcions de mutation
            # mutations = [self.__random_resetting_mutation, self.__swap_mutation, self.__partial_shuffle_mutation, self.__inversion_mutation, self.__displacement_mutation, self.__displacement_inversion_mutation]
            mutations = [self.__swap_mutation, self.__partial_shuffle_mutation, self.__inversion_mutation, self.__displacement_mutation, self.__displacement_inversion_mutation]
            # mutations = [self.__displacement_inversion_mutation]

            # Selecció de la funció de crossover aleatòria
            mutation_func = random.choice(mutations)
            c = mutation_func(child_vectors)
            return c, mutation_func.__name__
        else:
            return child_vectors, None

    # Funcions de mutation
    # def __random_resetting_mutation(self, child_vectors: Tuple[List[float], List[float]]) -> Tuple[List[float], List[float]]:
    def __random_resetting_mutation(self, child_vectors: Tuple[List[List[float]], List[List[float]]]) -> Tuple[List[List[float]], List[List[float]]]:
        # Capes per realitzar el crossover (les no seleccionades es passarà la capa sencera al fill 1 les del pare 1 i al fill 2 el del pare 2)
        total_selected_layers = random.randint(1, len(child_vectors[0]))
        available_layers_position = list(range(len(child_vectors[0])))
        selected_layers_position = []

        child_weights_vectors: List[List[float]] = []
        child_bias_vectors: List[List[float]] = []

        for _ in range(total_selected_layers):
            random_layer_position = random.choice(available_layers_position)
            selected_layers_position.append(random_layer_position)
            available_layers_position.remove(random_layer_position)

        for idx in range(len(child_vectors[0])):
            child_weights_vector: List[float] = child_vectors[0][idx]
            child_bias_vector: List[float] = child_vectors[1][idx]

            if idx not in selected_layers_position:
                child_weights_vectors.append(child_weights_vector)
                child_bias_vectors.append(child_bias_vector)
            else:
                # Obtenir un nombre aleatori de punts de mutació
                # print(round(len(child_weights_vector)), self.gens_mutation_ratio * 100)
                total_weight_mutation_points: int = random.randint(1, max(1, round(len(child_weights_vector) * self.gens_mutation_ratio)))
                total_bias_mutation_points: int = random.randint(1, max(1, round(len(child_bias_vector) * self.gens_mutation_ratio)))

                # print("total_weight_mutation_points", total_weight_mutation_points)
                # print("total_bias_mutation_points", total_bias_mutation_points)

                # Generar els punts de mutació únics entre 0 i el total de pesos del vector
                weight_slice_points: List[int] = sorted(random.sample(range(0, len(child_weights_vector)), total_weight_mutation_points))
                bias_slice_points: List[int] = sorted(random.sample(range(0, len(child_bias_vector)), total_bias_mutation_points))

                # print("weight_slice_points", weight_slice_points)
                # print("bias_slice_points", bias_slice_points)

                # Aplicar les mutacions per als pesos
                for weight_slice_point in weight_slice_points:
                    child_weights_vector[weight_slice_point] = random.uniform(self.min_valid_value, self.max_valid_value)

                for bias_slice_point in bias_slice_points:
                    child_bias_vector[bias_slice_point] = random.uniform(self.min_valid_value, self.max_valid_value)

                child_weights_vectors.append(child_weights_vector)
                child_bias_vectors.append(child_bias_vector)

        # print("child_weights_vector", child_1_weights)
        # print("child_bias", child_1_bias)

        return child_weights_vectors, child_bias_vectors

    def __swap_mutation(self, child_vectors: Tuple[List[float], List[float]]) -> Tuple[List[float], List[float]]:
        # Capes per realitzar el crossover (les no seleccionades es passarà la capa sencera al fill 1 les del pare 1 i al fill 2 el del pare 2)
        total_selected_layers = random.randint(1, len(child_vectors[0]))
        available_layers_position = list(range(len(child_vectors[0])))
        selected_layers_position = []

        child_weights_vectors: List[List[float]] = []
        child_bias_vectors: List[List[float]] = []

        for _ in range(total_selected_layers):
            random_layer_position = random.choice(available_layers_position)
            selected_layers_position.append(random_layer_position)
            available_layers_position.remove(random_layer_position)

        for idx in range(len(child_vectors[0])):
            child_weights_vector: List[float] = child_vectors[0][idx]
            child_bias_vector: List[float] = child_vectors[1][idx]

            if idx not in selected_layers_position:
                child_weights_vectors.append(child_weights_vector)
                child_bias_vectors.append(child_bias_vector)
            else:
                # Obtenir un nombre aleatori de punts de mutació
                total_weight_mutation_points: int = random.randint(1, max(1, round(len(child_weights_vector) * self.gens_mutation_ratio)))
                total_bias_mutation_points: int = random.randint(1, max(1, round(len(child_bias_vector) * self.gens_mutation_ratio)))

                # si el valor és imparell, se li suma 1
                total_weight_mutation_points += 1 if total_weight_mutation_points % 2 == 1 else 0
                total_bias_mutation_points += 1 if total_bias_mutation_points % 2 == 1 else 0

                # print("total_weight_mutation_points", total_weight_mutation_points)
                # print("total_bias_mutation_points", total_bias_mutation_points)

                # Generar els punts de mutació únics entre 0 i el total de pesos del vector
                weight_slice_points: List[int] = random.sample(range(0, len(child_weights_vector)), total_weight_mutation_points)
                bias_slice_points: List[int] = random.sample(range(0, len(child_bias_vector)), total_bias_mutation_points)

                # print("weight_slice_points", weight_slice_points)
                # print("bias_slice_points", bias_slice_points)

                # Aplicar les mutacions per als pesos
                for i in range(0, len(weight_slice_points), 2):
                    value_first_position = child_weights_vector[weight_slice_points[i]]
                    child_weights_vector[weight_slice_points[i]] = child_weights_vector[weight_slice_points[i + 1]]
                    child_weights_vector[weight_slice_points[i + 1]] = value_first_position

                for i in range(0, len(bias_slice_points), 2):
                    value_first_position = child_bias_vector[bias_slice_points[i]]
                    child_bias_vector[bias_slice_points[i]] = child_bias_vector[bias_slice_points[i + 1]]
                    child_bias_vector[bias_slice_points[i + 1]] = value_first_position

                child_weights_vectors.append(child_weights_vector)
                child_bias_vectors.append(child_bias_vector)

        # print("child_weights_vector", child_1_weights)
        # print("child_bias", child_1_bias)

        return child_weights_vectors, child_bias_vectors

    def __partial_shuffle_mutation(self, child_vectors: Tuple[List[float], List[float]]) -> Tuple[List[float], List[float]]:
        # Capes per realitzar el crossover (les no seleccionades es passarà la capa sencera al fill 1 les del pare 1 i al fill 2 el del pare 2)
        total_selected_layers = random.randint(1, len(child_vectors[0]))
        available_layers_position = list(range(len(child_vectors[0])))
        selected_layers_position = []

        child_weights_vectors: List[List[float]] = []
        child_bias_vectors: List[List[float]] = []

        for _ in range(total_selected_layers):
            random_layer_position = random.choice(available_layers_position)
            selected_layers_position.append(random_layer_position)
            available_layers_position.remove(random_layer_position)

        for idx in range(len(child_vectors[0])):
            child_weights_vector: List[float] = child_vectors[0][idx]
            child_bias_vector: List[float] = child_vectors[1][idx]

            if idx not in selected_layers_position:
                child_weights_vectors.append(child_weights_vector)
                child_bias_vectors.append(child_bias_vector)
            else:
                # Obtenir un nombre aleatori de gens a barallar
                total_weight_mutation_genes: int = random.randint(2, max(4, round(len(child_weights_vector) * self.gens_mutation_ratio)))
                total_bias_mutation_genes: int = random.randint(2, max(4, round(len(child_bias_vector) * self.gens_mutation_ratio)))

                # print("total_weight_mutation_genes", total_weight_mutation_genes)
                # print("total_bias_mutation_genes", total_bias_mutation_genes)

                # Generar els punts de mutació
                weights_starting_point: int = random.randint(0, len(child_weights_vector) - total_weight_mutation_genes)
                weights_ending_point: int = weights_starting_point + total_weight_mutation_genes
                bias_starting_point: int = random.randint(0, len(child_bias_vector) - total_bias_mutation_genes)
                bias_ending_point: int = bias_starting_point + total_bias_mutation_genes

                # print("weights_starting_point", weights_starting_point)
                # print("weights_ending_point", weights_ending_point)

                # Extreure la subllista
                changing_weights = child_weights_vector[weights_starting_point:weights_ending_point]
                changing_bias = child_bias_vector[bias_starting_point:bias_ending_point]

                # Barallar subllistes
                random.shuffle(changing_weights)
                random.shuffle(changing_bias)

                child_weights_vector[weights_starting_point:weights_ending_point] = changing_weights
                child_bias_vector[bias_starting_point:bias_ending_point] = changing_bias

                child_weights_vectors.append(child_weights_vector)
                child_bias_vectors.append(child_bias_vector)

        # print("child_weights_vector", child_1_weights)
        # print("child_bias", child_1_bias)

        return child_weights_vectors, child_bias_vectors

    def __inversion_mutation(self, child_vectors: Tuple[List[float], List[float]]) -> Tuple[List[float], List[float]]:
        # Capes per realitzar el crossover (les no seleccionades es passarà la capa sencera al fill 1 les del pare 1 i al fill 2 el del pare 2)
        total_selected_layers = random.randint(1, len(child_vectors[0]))
        available_layers_position = list(range(len(child_vectors[0])))
        selected_layers_position = []

        child_weights_vectors: List[List[float]] = []
        child_bias_vectors: List[List[float]] = []

        for _ in range(total_selected_layers):
            random_layer_position = random.choice(available_layers_position)
            selected_layers_position.append(random_layer_position)
            available_layers_position.remove(random_layer_position)

        for idx in range(len(child_vectors[0])):
            child_weights_vector: List[float] = child_vectors[0][idx]
            child_bias_vector: List[float] = child_vectors[1][idx]

            if idx not in selected_layers_position:
                child_weights_vectors.append(child_weights_vector)
                child_bias_vectors.append(child_bias_vector)
            else:
                # Obtenir un nombre aleatori de gens a invertir
                total_weight_mutation_genes: int = random.randint(2, max(4, round(len(child_weights_vector) * self.gens_mutation_ratio)))
                total_bias_mutation_genes: int = random.randint(2, max(4, round(len(child_bias_vector) * self.gens_mutation_ratio)))

                # print("total_weight_mutation_genes", total_weight_mutation_genes)
                # print("total_bias_mutation_genes", total_bias_mutation_genes)

                # Generar els punts de mutació
                weights_starting_point: int = random.randint(0, len(child_weights_vector) - total_weight_mutation_genes)
                weights_ending_point: int = weights_starting_point + total_weight_mutation_genes
                bias_starting_point: int = random.randint(0, len(child_bias_vector) - total_bias_mutation_genes)
                bias_ending_point: int = bias_starting_point + total_bias_mutation_genes

                # print("weights_starting_point", weights_starting_point)
                # print("weights_ending_point", weights_ending_point)

                # Extreure la subllista i invertir els elements
                changing_weights = child_weights_vector[weights_starting_point:weights_ending_point][::-1]
                changing_bias = child_bias_vector[bias_starting_point:bias_ending_point][::-1]

                child_weights_vector[weights_starting_point:weights_ending_point] = changing_weights
                child_bias_vector[bias_starting_point:bias_ending_point] = changing_bias

                child_weights_vectors.append(child_weights_vector)
                child_bias_vectors.append(child_bias_vector)

        # print("child_weights_vector", child_1_weights)
        # print("child_bias", child_1_bias)

        return child_weights_vectors, child_bias_vectors

    def __displacement_mutation(self, child_vectors: Tuple[List[float], List[float]]) -> Tuple[List[float], List[float]]:
        # Capes per realitzar el crossover (les no seleccionades es passarà la capa sencera al fill 1 les del pare 1 i al fill 2 el del pare 2)
        total_selected_layers = random.randint(1, len(child_vectors[0]))
        available_layers_position = list(range(len(child_vectors[0])))
        selected_layers_position = []

        child_weights_vectors: List[List[float]] = []
        child_bias_vectors: List[List[float]] = []

        for _ in range(total_selected_layers):
            random_layer_position = random.choice(available_layers_position)
            selected_layers_position.append(random_layer_position)
            available_layers_position.remove(random_layer_position)

        for idx in range(len(child_vectors[0])):
            child_weights_vector: List[float] = child_vectors[0][idx]
            child_bias_vector: List[float] = child_vectors[1][idx]

            if idx not in selected_layers_position:
                child_weights_vectors.append(child_weights_vector)
                child_bias_vectors.append(child_bias_vector)
            else:
                # Obtenir un nombre aleatori de gens per canviar de posició
                total_weight_mutation_genes: int = random.randint(2, max(4, round(len(child_weights_vector) * self.gens_mutation_ratio)))
                total_bias_mutation_genes: int = random.randint(2, max(4, round(len(child_bias_vector) * self.gens_mutation_ratio)))

                # print("total_weight_mutation_genes", total_weight_mutation_genes)
                # print("total_bias_mutation_genes", total_bias_mutation_genes)

                # Generar els punts de mutació
                weights_starting_point: int = random.randint(0, len(child_weights_vector) - total_weight_mutation_genes)
                weights_ending_point: int = weights_starting_point + total_weight_mutation_genes
                bias_starting_point: int = random.randint(0, len(child_bias_vector) - total_bias_mutation_genes)
                bias_ending_point: int = bias_starting_point + total_bias_mutation_genes

                # print("weights_starting_point", weights_starting_point)
                # print("weights_ending_point", weights_ending_point)

                # Extreure la subllista
                changing_weights = child_weights_vector[weights_starting_point:weights_ending_point]
                changing_bias = child_bias_vector[bias_starting_point:bias_ending_point]

                # Eliminar la subllista original
                del child_weights_vector[weights_starting_point:weights_ending_point]
                del child_bias_vector[bias_starting_point:bias_ending_point]

                # Escollir un índex de destí aleatori
                index_weights = random.randint(0, len(child_weights_vector))
                index_bias = random.randint(0, len(child_bias_vector))

                # Insertar la subllista a la nova posició
                child_weights_vector[index_weights:index_weights] = changing_weights
                child_bias_vector[index_bias:index_bias] = changing_bias

                child_weights_vectors.append(child_weights_vector)
                child_bias_vectors.append(child_bias_vector)

        # print("child_weights_vector", child_1_weights)
        # print("child_bias", child_1_bias)

        return child_weights_vectors, child_bias_vectors

    def __displacement_inversion_mutation(self, child_vectors: Tuple[List[float], List[float]]) -> Tuple[List[float], List[float]]:
        # Capes per realitzar el crossover (les no seleccionades es passarà la capa sencera al fill 1 les del pare 1 i al fill 2 el del pare 2)
        total_selected_layers = random.randint(1, len(child_vectors[0]))
        available_layers_position = list(range(len(child_vectors[0])))
        selected_layers_position = []

        child_weights_vectors: List[List[float]] = []
        child_bias_vectors: List[List[float]] = []

        for _ in range(total_selected_layers):
            random_layer_position = random.choice(available_layers_position)
            selected_layers_position.append(random_layer_position)
            available_layers_position.remove(random_layer_position)

        for idx in range(len(child_vectors[0])):
            child_weights_vector: List[float] = child_vectors[0][idx]
            child_bias_vector: List[float] = child_vectors[1][idx]

            if idx not in selected_layers_position:
                child_weights_vectors.append(child_weights_vector)
                child_bias_vectors.append(child_bias_vector)
            else:
                # Obtenir un nombre aleatori de gens per canviar de posició
                total_weight_mutation_genes: int = random.randint(2, max(4, round(len(child_weights_vector) * self.gens_mutation_ratio)))
                total_bias_mutation_genes: int = random.randint(2, max(4, round(len(child_bias_vector) * self.gens_mutation_ratio)))

                # print("total_weight_mutation_genes", total_weight_mutation_genes)
                # print("total_bias_mutation_genes", total_bias_mutation_genes)

                # Generar els punts de mutació
                weights_starting_point: int = random.randint(0, len(child_weights_vector) - total_weight_mutation_genes)
                weights_ending_point: int = weights_starting_point + total_weight_mutation_genes
                bias_starting_point: int = random.randint(0, len(child_bias_vector) - total_bias_mutation_genes)
                bias_ending_point: int = bias_starting_point + total_bias_mutation_genes

                # print("weights_starting_point", weights_starting_point)
                # print("weights_ending_point", weights_ending_point)

                # Extreure la subllista i invertir-la
                changing_weights = child_weights_vector[weights_starting_point:weights_ending_point][::-1]
                changing_bias = child_bias_vector[bias_starting_point:bias_ending_point][::-1]

                # Eliminar la subllista original
                del child_weights_vector[weights_starting_point:weights_ending_point]
                del child_bias_vector[bias_starting_point:bias_ending_point]

                # Escollir un índex de destí aleatori
                index_weights = random.randint(0, len(child_weights_vector))
                index_bias = random.randint(0, len(child_bias_vector))

                # Insertar la subllista a la nova posició
                child_weights_vector[index_weights:index_weights] = changing_weights
                child_bias_vector[index_bias:index_bias] = changing_bias

                child_weights_vectors.append(child_weights_vector)
                child_bias_vectors.append(child_bias_vector)

        # print("child_weights_vector", child_1_weights)
        # print("child_bias", child_1_bias)

        return child_weights_vectors, child_bias_vectors
