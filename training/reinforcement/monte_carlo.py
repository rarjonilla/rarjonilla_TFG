import os
import pickle
from copy import deepcopy
from typing import Tuple, Dict, List
import numpy as np


class Monte_carlo:

    def __init__(self, eps: float, eps_decrease: float, gamma: float, model_type: int, model_path: str) -> None:
        self.__model_type: int = model_type
        self.__model_path: str = model_path

        # Probabilitat d'exploració
        # self.__eps: float = 0.05
        self.__eps: float = eps
        self.__eps_decrease: float = eps_decrease

        # Valor de Discount (1 = undiscount)
        # self.__gamma: float = 1.0
        self.__gamma: float = gamma

        # Valor esperat de les accions (Agent estimated future reward)
        self.__q: Dict = {}

        # Llista de possibles accions
        self.__action_space: List[int] = []
        self.__actions: Dict = {}

        # TODO -> returns i Q és el mateix. Només cal 1 dels 2 (estalvio memoria RAM, recursos d'emmagatzematge i temps de guardar / carregar)
        # Diccionari que conté els returns (G) de cada estat
        # self.__returns: Dict = {}

        # Diccionari que conté les vegades que s'ha visitat un estat i acció concret
        self.__pairs_visited: Dict = {}

        # La política és l'estratègia de l'agent
        self.__policy: Dict = {}

        # Llista que conte els estats, accions i returns (G) al llarg de la partida
        self.__states_actions_returns: List[Tuple[int, int, float]] = []

        # Llista que conte els estats, accions i reward de cada ronda al llarg de la partida# Llista que conte els estats, accions i reward de cada ronda al llarg de la partida
        self.__memories: List[Tuple[int, int, int]] = []

        # self.__last_memory: Tuple[int, int, int] = (0, 0, 0)
        self.__state: int = 0
        self.__reward: int = 0
        self.__action: int = 0

        self.load_model()

    def del_model(self):
        self.__q = None
        self.__pairs_visited = None
        self.__policy = None
        self.__states_actions_returns = None
        self.__memories = None


    def load_model(self) -> None:
        # Comprovem si ja existeix
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

            # with open(self.__model_path + "/returns.pkl", 'rb') as returns_file:
            #     self.__returns = pickle.load(returns_file)

            with open(self.__model_path + "/info.pkl", 'rb') as info_file:
                info: Dict = pickle.load(info_file)
                self.__eps = info["eps"]
                self.__eps_decrease = info["eps_decrease"]
                self.__gamma = info["gamma"]

                print(self.__eps)

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

        # with open(self.__model_path + "/returns.pkl", 'wb') as returns_file:
        #     pickle.dump(self.__returns, returns_file)

        info: Dict = {
            'eps': self.__eps,
            'eps_decrease': self.__eps_decrease,
            'gamma': self.__gamma,
        }

        with open(self.__model_path + "/info.pkl", 'wb') as info_file:
            pickle.dump(info, info_file)

    def add_memory(self, player_id: int):
        self.__memories.append((self.__state, self.__action, self.__reward, deepcopy(self.__action_space)))

    def set_reward(self, reward: int, player_id: int):
        self.__reward = reward

    def set_state(self, state: int, player_id: int):
        self.__state = state

    def choose_action_from_policy(self, player_id: int) -> int:
        # Es tria u na acció del policy (si no existeix es tria aleatoriament)
        if self.__state in self.__policy:
            # TODO -> haig de triar tenint en compte les accions disponibles
            print("policy")
            self.__action = self.__policy[self.__state]
            if self.__action not in self.__action_space:
                # print("!!!!!")
                # Obtener la representación binaria como cadena de caracteres y eliminar el prefijo '0b'
                cadena_binaria = bin(self.__state)[2:]

                # Convertir la cadena binaria a una lista de bits
                lista_binaria = [int(bit) for bit in cadena_binaria]
                # print("Lista binaria:", lista_binaria)
        else:
            print("random")
            self.__action = np.random.choice(self.__action_space)

        return self.__action

    def new_episode(self):
        self.__states_actions_returns = []
        self.__memories = []

    def set_action_space(self, action_space, player_id: int):
        self.__action_space = action_space

        if self.__state not in self.__actions:
            self.__actions[self.__state] = action_space

    def __decrease_eps(self):
        # A cada partida que passa, la taxa d'exploració va perdent força
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
                # S'afegeix a la llista que conte els estats, accions i returns al llarg de la partida
                self.__states_actions_returns.append((state, action, g, available_actions))

            g = self.__gamma * g + reward

        # Falta una?
        if state is not None:
            self.__states_actions_returns.append((state, action, g, available_actions))

        # Es capgira la llista per tenir-la en ordre d'ocurrencia (de principi del joc a final)
        self.__states_actions_returns.reverse()

    def update_policy(self):
        self.__calculate_returns()

        # Llista per guardar els espais i accions vistes a la partida
        states_actions_visited = []

        # Es recorre la llista d'estats, accions i retorns en ordre de la partida
        for state, action, g, available_actions in self.__states_actions_returns:
            # Parell (estat - acció)
            sa = (state, action)
            # En principi mai es repeteix. Estem parlant d'una mateixa partida
            if sa not in states_actions_visited:
                # Si existeix l'actualitzo, sino el creo a 1
                if sa in self.__pairs_visited:
                    self.__pairs_visited[sa] += 1
                else:
                    self.__pairs_visited[sa] = 1

                # incremental implementation -> amb això s'obté la mitjana dels returns per a un estat concret
                #   sense haver de calcular cada cop la mitjana amb tots els results anteriors.
                # https://www.analyticsvidhya.com/blog/2018/11/reinforcement-learning-introduction-monte-carlo-learning-openai-gym/
                # Every visit monte carlo

                # new estimate = 1 / N * [sample - old estimate]
                # Es calcula la nova estimació per aquest parell (estat - acció)
                # if sa in self.__returns:
                if sa in self.__q:
                    # self.__returns[sa] += (1 / self.__pairs_visited[sa]) * (g - self.__returns[sa])
                    self.__q[sa] += (1 / self.__pairs_visited[sa]) * (g - self.__q[sa])
                else:
                    # self.__returns[sa] = (1 / self.__pairs_visited[sa]) * g
                    self.__q[sa] = (1 / self.__pairs_visited[sa]) * g

                # S'actualitza l'estimació de l'estat acció (jo afegire o actualitzare segons si el tinc o no)
                # self.__q[sa] = self.__returns[sa]

                # Actualització de la política
                rand = np.random.random()
                if rand < 1 - self.__eps:
                    # Es trien les millors accions per aquest estat (en cas d'empat s'en selecciones totes les iguals)
                    # Jo tindre les meves possibles accions (cartes a la mà i canvi) en comptes de "actionSpace"
                    values = []
                    # for a in self.__action_space:
                    for a in available_actions:
                        if (state, a) in self.__q:
                            values.append(self.__q[(state, a)])
                    values = np.array(values)
                    # values = np.array([Q[(state, a)] for a in actionSpace])
                    # En cas d'empat, es tria una aleatoria
                    # if len(values) > 0:
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
                        # self.__policy[state] = self.__action_space[best]
                        self.__policy[state] = available_actions[best]
                else:
                    # S'actualitza la política amb una acció aleatoria
                    # TODO -> Jo tindre només les accions possibles
                    # self.__policy[state] = np.random.choice(self.__action_space)
                    self.__policy[state] = np.random.choice(available_actions)

                states_actions_visited.append(sa)

        self.__decrease_eps()
