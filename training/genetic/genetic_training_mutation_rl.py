import random
from typing import Tuple, List, Optional, Dict


class Genetic_training_mutation_rl:
    """Classe de Funcions de mutació per RL"""
    def __init__(self, mutation_ratio: float):
        # Probabilitat de mutació
        self.__mutation_ratio: float = mutation_ratio
        # Nota: poso un limit d'un 5% dels gens per a la mutació (és probable que aquest valor varii al fer proves)
        self.gens_mutation_ratio: float = 0.05

    def update_mutation_ratio(self, mutation_ratio: float):
        self.__mutation_ratio = mutation_ratio

    def mutation(self, child_params: Tuple[Dict, Dict, Dict, Dict]) -> Tuple[Tuple[Dict, Dict, Dict, Dict], Optional[str]]:
        # S'executa la mutació si l'atzar ho vol
        if random.random() <= self.__mutation_ratio:
            # LLista de funcions de mutation
            mutations = [self.__random_resetting_mutation]

            # Selecció de la funció de crossover aleatòria
            mutation_func = random.choice(mutations)

            # Execució de la funció de mutació
            c = mutation_func(child_params)
            return c, mutation_func.__name__
        else:
            return child_params, None

    # Funcions de mutation
    def __random_resetting_mutation(self, child_params: Tuple[Dict, Dict, Dict, Dict]) -> Tuple[Dict, Dict, Dict, Dict]:
        q_c1, pairs_visited_c1, policy_c1, actions_c1 = child_params

        # Obtenir un nombre aleatori de punts de mutació
        policy_keys = policy_c1.keys()
        total_mutation_points: int = random.randint(1, max(1, round(len(policy_keys) * self.gens_mutation_ratio)))

        for _ in range(total_mutation_points):
            # Seleccionar una clau aleatoria per mutar
            key_to_mutate = random.choice(list(policy_keys))
            # Generar una nova acció d'entre les possibles
            new_action = random.choice(actions_c1[key_to_mutate])
            policy_c1[key_to_mutate] = new_action

        return q_c1, pairs_visited_c1, policy_c1, actions_c1
