import os
import pickle
from copy import deepcopy
from typing import Tuple, Dict, List
import numpy as np


class Monte_carlo:
    """Classe Monte Carlo - 1 estat"""

    def __init__(self, eps: float, eps_decrease: float, gamma: float, model_type: int, model_path: str) -> None:
        # Informació dels models
        self.__model_type: int = model_type
        self.__model_path: str = model_path

        # Probabilitat d'exploració
        self.__eps: float = eps
        # Decreixement d'exploració
        self.__eps_decrease: float = eps_decrease

        # Valor de descompte
        self.__gamma: float = gamma

        # Valor esperat de les accions (Agent estimated future reward)
        self.__q: Dict = {}

        # Llista de possibles accions
        self.__action_space: List[int] = []

        # TODO s'ha eliminat de l'esta múltiple, no cal
        # Diccionari d'accions (eliminat de )
        self.__actions: Dict = {}

        # Diccionari que conté les vegades que s'ha visitat un estat i acció concret
        self.__pairs_visited: Dict = {}

        # La política és l'estratègia de l'agent
        self.__policy: Dict = {}

        # Llista que conté els estats, accions i returns (G) al llarg de la partida
        self.__states_actions_returns: List[Tuple[int, int, float]] = []

        # Llista que conté els estats, accions i reward de cada ronda al llarg de la partida# Llista que conte els estats, accions i reward de cada ronda al llarg de la partida
        self.__memories: List[Tuple[int, int, int]] = []

        # Estat actual
        self.__state: int = 0
        # Recompensa actual
        self.__reward: int = 0
        # Acció actual
        self.__action: int = 0

        self.load_model()

    def del_model(self):
        self.__q = None
        self.__pairs_visited = None
        self.__policy = None
        self.__states_actions_returns = None
        self.__memories = None

    def load_model(self) -> None:
        # Comprovem si ja existeix l'arxiu (s'ha guardat almenys un cop) per carregar els diccionaris i seguir l'entrenament
        if os.path.exists(self.__model_path + "/q.pkl"):
            with open(self.__model_path + "/q.pkl", 'rb') as q_file:
                self.__q = pickle.load(q_file)

            with open(self.__model_path + "/pairs_visited.pkl", 'rb') as pairs_visites_file:
                self.__pairs_visited = pickle.load(pairs_visites_file)

            with open(self.__model_path + "/policy.pkl", 'rb') as policy_file:
                self.__policy = pickle.load(policy_file)

            if os.path.exists(self.__model_path + "/actions.pkl"):
                with open(self.__model_path + "/actions.pkl", 'rb') as actions_file:
                    self.__actions = pickle.load(actions_file)

            with open(self.__model_path + "/info.pkl", 'rb') as info_file:
                info: Dict = pickle.load(info_file)
                self.__eps = info["eps"]
                self.__eps_decrease = info["eps_decrease"]
                self.__gamma = info["gamma"]

    def save_model(self) -> None:
        if not os.path.exists(self.__model_path):
            os.makedirs(self.__model_path)

        # TODO -> pairs visited i Q comparteixen les keys. Puc fer un sol fitxer per no malgastar espai?
        with open(self.__model_path + "/q.pkl", 'wb') as q_file:
            pickle.dump(self.__q, q_file)

        with open(self.__model_path + "/pairs_visited.pkl", 'wb') as pairs_visites_file:
            pickle.dump(self.__pairs_visited, pairs_visites_file)

        with open(self.__model_path + "/policy.pkl", 'wb') as policy_file:
            pickle.dump(self.__policy, policy_file)

        with open(self.__model_path + "/actions.pkl", 'wb') as actions_file:
            pickle.dump(self.__actions, actions_file)

        info: Dict = {
            'eps': self.__eps,
            'eps_decrease': self.__eps_decrease,
            'gamma': self.__gamma,
        }

        with open(self.__model_path + "/info.pkl", 'wb') as info_file:
            pickle.dump(info, info_file)

    def add_memory(self, player_id: int):
        # Afegir memòria a la llista
        self.__memories.append((self.__state, self.__action, self.__reward, deepcopy(self.__action_space)))

    def set_reward(self, reward: int, player_id: int):
        # S'indica la recompensa rebuda
        self.__reward = reward

    def set_state(self, state: int, player_id: int):
        # S'indica l'estat actual
        self.__state = state

    def choose_action_from_policy(self, player_id: int) -> int:
        # Es tria una acció del policy (si no existeix es tria aleatoriament)
        if self.__state in self.__policy:
            self.__action = self.__policy[self.__state]
        else:
            self.__action = np.random.choice(self.__action_space)

        return self.__action

    def new_episode(self):
        # Iniciar nou episodi
        self.__states_actions_returns = []
        self.__memories = []

    def set_action_space(self, action_space, player_id: int):
        # S'indica les possibles accions que es poden realitzar
        self.__action_space = action_space

        if self.__state not in self.__actions:
            self.__actions[self.__state] = action_space

    def __decrease_eps(self):
        # A cada episodi que passa, la taxa d'exploració va perdent força
        self.__eps -= self.__eps_decrease if self.__eps - self.__eps_decrease > 0 else 0

    def __calculate_returns(self):
        # Retorn total acumulat al llarg de la partida
        g = 0
        last = True
        state = None

        # Es recorre la memoria a la inversa (del final a l'inici)
        for state, action, reward, available_actions in reversed(self.__memories):
            # L'acció final no s'afegeix a la llista, encara s'ha de calcular
            if last:
                last = False
            else:
                # S'afegeix a la llista que conté els estats, accions i returns al llarg de la partida
                self.__states_actions_returns.append((state, action, g, available_actions))

            g = self.__gamma * g + reward

        # Falta afegir la primera (l'última que es calcula)
        if state is not None:
            self.__states_actions_returns.append((state, action, g, available_actions))

        # Es capgira la llista per tenir-la en ordre d'ocurrència (de principi del joc a final)
        self.__states_actions_returns.reverse()

    def update_policy(self):
        # Actualització de la política
        # Calcul dels retorns
        self.__calculate_returns()

        # Llista per guardar els espais i accions vistes a la partida
        states_actions_visited = []

        # Es recorre la llista d'estats, accions i retorns en ordre de la partida
        for state, action, g, available_actions in self.__states_actions_returns:
            # Parell (estat - acció)
            sa = (state, action)

            # TODO - En principi mai es repeteix un mateix estat durant el mateix episodi. Estem parlant d'una mateixa partida. Es podria treure aquesta condició i hauria de funcionar igual
            if sa not in states_actions_visited:
                # Si existeix l'actualitzo, sino el creo a 1
                if sa in self.__pairs_visited:
                    self.__pairs_visited[sa] += 1
                else:
                    self.__pairs_visited[sa] = 1

                # incremental implementation -> amb això s'obté la mitjana dels returns per a un estat concret sense haver de calcular cada cop la mitjana amb tots els results anteriors.
                # https://www.analyticsvidhya.com/blog/2018/11/reinforcement-learning-introduction-monte-carlo-learning-openai-gym/
                # Every visit monte carlo
                # new estimate = 1 / N * [sample - old estimate]
                # Es calcula la nova estimació per aquest parell (estat - acció)
                if sa in self.__q:
                    self.__q[sa] += (1 / self.__pairs_visited[sa]) * (g - self.__q[sa])
                else:
                    self.__q[sa] = (1 / self.__pairs_visited[sa]) * g

                # Actualització de la política
                rand = np.random.random()
                if rand < 1 - self.__eps:
                    # Es trien les millors accions per aquest estat
                    values = []
                    for a in available_actions:
                        if (state, a) in self.__q:
                            values.append(self.__q[(state, a)])
                    values = np.array(values)

                    # En cas d'empat es tria una aleatòria
                    best = np.random.choice(np.where(values == values.max())[0])

                    if values[best] < 0 and len(values) < len(available_actions):
                        # Aquest cas contempla que, si el valor esperat és negatiu i no s'han visitat totes les accions,
                        # S'escollirà una de les accions restants aleatòries per afegir a la policy
                        # Representa que s'ha d'inicialitzar els valors de Q a 0 al principi de l'entrenament
                        # Com que jo no els tinc, si el primer cop perd 20 punts, ell sempre escolliria aquesta, ja que la resta de valors no hi son
                        # Així representa que estaria agafant un dels altres valors que estan a 0, que serien millors que aquesta opció de -20

                        other_available_actions = [action for action in available_actions if action != available_actions[best]]
                        self.__policy[state] = np.random.choice(other_available_actions)
                    else:
                        # S'actualitza la política amb la millor acció
                        self.__policy[state] = available_actions[best]
                else:
                    # S'actualitza la política amb una acció aleatoria
                    self.__policy[state] = np.random.choice(available_actions)

                states_actions_visited.append(sa)

        self.__decrease_eps()
