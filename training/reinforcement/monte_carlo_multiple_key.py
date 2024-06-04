import os
import pickle
from copy import deepcopy
from typing import Tuple, Dict, List
import numpy as np
from collections import Counter



class Monte_carlo_multiple_state:

    def __init__(self, eps: float, eps_decrease: float, gamma: float, model_type: int, model_path: str, is_brisca: bool, is_training: bool = False) -> None:
        self.__model_type: int = model_type
        self.__model_path: str = model_path
        self.__is_brisca = is_brisca
        self.__is_training = is_training

        # Probabilitat d'exploració
        # self.__eps: float = 0.05
        self.__eps: float = eps
        self.__eps_decrease: float = eps_decrease

        # Valor de Discount (1 = undiscount)
        # self.__gamma: float = 1.0
        self.__gamma: float = gamma

        # Valor esperat de les accions (Agent estimated future reward)
        # self.__q: Dict = {}
        self.__q_pairs_visited: Dict = {}

        # Llista de possibles accions
        self.__action_space_p1: List[List[int]] = []
        self.__action_space_p2: List[List[int]] = []
        self.__action_space_p3: List[List[int]] = []
        self.__action_space_p4: List[List[int]] = []
        # self.__actions: Dict = {}

        # Diccionari que conté els returns (G) de cada estat
        # self.__returns: Dict = {}

        # Diccionari que conté les vegades que s'ha visitat un estat i acció concret
        # self.__pairs_visited: Dict = {}

        # La política és l'estratègia de l'agent
        # self.__policy: Dict = {}
        self.__policy: Dict = {}

        # Llista que conte els estats, accions i returns (G) al llarg de la partida
        self.__states_actions_returns_p1: List[Tuple[int, int, float]] = []
        self.__states_actions_returns_p2: List[Tuple[int, int, float]] = []
        self.__states_actions_returns_p3: List[Tuple[int, int, float]] = []
        self.__states_actions_returns_p4: List[Tuple[int, int, float]] = []

        # Llista que conte els estats, accions i reward de cada ronda al llarg de la partida# Llista que conte els estats, accions i reward de cada ronda al llarg de la partida
        self.__memories_p1: List[Tuple[List[int], int, int]] = []
        self.__memories_p2: List[Tuple[List[int], int, int]] = []
        self.__memories_p3: List[Tuple[List[int], int, int]] = []
        self.__memories_p4: List[Tuple[List[int], int, int]] = []

        # self.__last_memory: Tuple[int, int, int] = (0, 0, 0)
        # Tenim diferents estats:
        #  - estat carta de triomf
        #  - estat cartes jugades
        #  - estat cartes a la mà
        #  - estat cartes vistes
        #  - regles
        self.__state_p1: List[int] = 0
        self.__state_p2: List[int] = 0
        self.__state_p3: List[int] = 0
        self.__state_p4: List[int] = 0

        self.__reward_p1: int = 0
        self.__reward_p2: int = 0
        self.__reward_p3: int = 0
        self.__reward_p4: int = 0

        self.__action_p1: int = 0
        self.__action_p2: int = 0
        self.__action_p3: int = 0
        self.__action_p4: int = 0

        self.load_model()

    def del_model(self):
        # self.__q = None
        # self.__pairs_visited = None
        # self.__policy = None
        # self.__actions = None
        self.__q_pairs_visited = None
        self.__policy = None

        self.__states_actions_returns_p1 = None
        self.__states_actions_returns_p2 = None
        self.__states_actions_returns_p3 = None
        self.__states_actions_returns_p4 = None
        self.__memories_p1 = None
        self.__memories_p2 = None
        self.__memories_p3 = None
        self.__memories_p4 = None

    def load_model(self) -> None:
        # Comprovem si ja existeix
        if os.path.exists(self.__model_path + "/policy.pkl"):
            if self.__is_training:
                with open(self.__model_path + "/q_pv.pkl", 'rb') as q_pv_file:
                    self.__q_pairs_visited = pickle.load(q_pv_file)

            with open(self.__model_path + "/policy.pkl", 'rb') as policy_file:
                self.__policy = pickle.load(policy_file)

            with open(self.__model_path + "/info.pkl", 'rb') as info_file:
                info: Dict = pickle.load(info_file)
                self.__eps = info["eps"]
                self.__eps_decrease = info["eps_decrease"]
                self.__gamma = info["gamma"]

                print(self.__eps)
                print("mc multiple!")

    def save_model(self) -> None:
        if not os.path.exists(self.__model_path):
            os.makedirs(self.__model_path)

        # TODO -> pairs visited i Q comparteixen les keys. Puc fer un sol fitxer per no malgastar espai?
        with open(self.__model_path + "/q_pv.pkl", 'wb') as q_pv_file:
            pickle.dump(self.__q_pairs_visited, q_pv_file)

        with open(self.__model_path + "/policy.pkl", 'wb') as policy_file:
            pickle.dump(self.__policy, policy_file)

        info: Dict = {
            'eps': self.__eps,
            'eps_decrease': self.__eps_decrease,
            'gamma': self.__gamma,
        }

        with open(self.__model_path + "/info.pkl", 'wb') as info_file:
            pickle.dump(info, info_file)

    def add_memory(self, player_id: int):
        if player_id == 0:
            self.__memories_p1.append((self.__state_p1, self.__action_p1, self.__reward_p1, deepcopy(self.__action_space_p1)))
        elif player_id == 1:
            self.__memories_p2.append((self.__state_p2, self.__action_p2, self.__reward_p2, deepcopy(self.__action_space_p2)))
        elif player_id == 2:
            self.__memories_p3.append((self.__state_p3, self.__action_p3, self.__reward_p3, deepcopy(self.__action_space_p3)))
        elif player_id == 3:
            self.__memories_p4.append((self.__state_p4, self.__action_p4, self.__reward_p4, deepcopy(self.__action_space_p4)))

    def set_reward(self, reward: int, player_id: int):
        if player_id == 0:
            self.__reward_p1 = reward
        elif player_id == 1:
            self.__reward_p2 = reward
        elif player_id == 2:
            self.__reward_p3 = reward
        elif player_id == 3:
            self.__reward_p4 = reward

    def set_state(self, state: List[int], player_id: int):
        if player_id == 0:
            self.__state_p1 = state
        elif player_id == 1:
            self.__state_p2 = state
        elif player_id == 2:
            self.__state_p3 = state
        elif player_id == 3:
            self.__state_p4 = state

    def choose_action_from_policy(self, player_id: int) -> int:
        # Es tria u na acció del policy (si no existeix es tria aleatoriament)
        state = self.__state_p1

        if player_id == 1:
            state = self.__state_p2
        elif player_id == 2:
            state = self.__state_p3
        elif player_id == 3:
            state = self.__state_p4

        if state in self.__policy:
            # TODO -> haig de triar tenint en compte les accions disponibles
            if player_id == 0:
                self.__action_p1 = self.__policy[state]
            elif player_id == 1:
                self.__action_p2 = self.__policy[state]
            elif player_id == 2:
                self.__action_p3 = self.__policy[state]
            elif player_id == 3:
                self.__action_p4 = self.__policy[state]

            # if self.__action not in self.__action_space:
            # print("!!!!!")
            # Obtener la representación binaria como cadena de caracteres y eliminar el prefijo '0b'
            # cadena_binaria = bin(self.__state)[2:]

            # Convertir la cadena binaria a una lista de bits
            # lista_binaria = [int(bit) for bit in cadena_binaria]
            # print("Lista binaria:", lista_binaria)
        else:
            if self.__is_training:
                if player_id == 0:
                    self.__action_p1 = np.random.choice(self.__action_space_p1)
                elif player_id == 1:
                    self.__action_p2 = np.random.choice(self.__action_space_p2)
                elif player_id == 2:
                    self.__action_p3 = np.random.choice(self.__action_space_p3)
                elif player_id == 3:
                    self.__action_p4 = np.random.choice(self.__action_space_p4)
            else:
                # Tenim diferents estats:
                #  - estat carta de triomf
                #  - estat cartes jugades
                #  - estat cartes a la mà
                #  - estat de cants (Tute)
                #  - estat cartes vistes
                #  - estat regles

                if player_id == 0:
                    trump_suit = self.__state_p1[0]
                    # trump_card = self.__state_p1[1]
                    played_cards_state = self.__state_p1[1]
                    hand_cards_state = self.__state_p1[2]
                    singed_state = self.__state_p1[3]
                    viewed_cards_state = self.__state_p1[4]
                    rules_state = self.__state_p1[5]
                elif player_id == 1:
                    trump_suit = self.__state_p2[0]
                    # trump_card = self.__state_p2[1]
                    played_cards_state = self.__state_p2[1]
                    hand_cards_state = self.__state_p2[2]
                    singed_state = self.__state_p2[3]
                    viewed_cards_state = self.__state_p2[4]
                    rules_state = self.__state_p2[5]
                elif player_id == 2:
                    trump_suit = self.__state_p3[0]
                    # trump_card = self.__state_p3[1]
                    played_cards_state = self.__state_p3[1]
                    hand_cards_state = self.__state_p3[2]
                    singed_state = self.__state_p3[3]
                    viewed_cards_state = self.__state_p3[4]
                    rules_state = self.__state_p3[5]
                elif player_id == 3:
                    trump_suit = self.__state_p4[0]
                    # trump_card = self.__state_p4[1]
                    played_cards_state = self.__state_p4[1]
                    hand_cards_state = self.__state_p4[2]
                    singed_state = self.__state_p4[3]
                    viewed_cards_state = self.__state_p4[4]
                    rules_state = self.__state_p4[5]

                if not self.__is_brisca:
                    # Buscarem si hem vist situacions semblants per no donar una acció aleatòria
                    # Es prova amb qualsevol carta de triomf del pal corresponent (s'elimina el valor de la carta)
                    similar_keys = {key for key in self.__policy.keys() if trump_suit == key[0] and played_cards_state == key[1] and hand_cards_state == key[2] and singed_state == key[3] and viewed_cards_state == key[4] and rules_state == key[5]}
                    # print("choose 0")

                    #                     if len(similar_keys) == 0:
                    #                         # Es treuen les regles i el valor de la carta de triomf
                    #                         similar_keys = {key for key in self.__policy.keys() if trump_suit == key[0] and played_cards_state == key[1] and hand_cards_state == key[2] and singed_state == key[3] and viewed_cards_state == key[4]}
                    #                         # print("choose 1")
                    #
                    #                     if len(similar_keys) == 0:
                    #                         # Es treuen les cartes vistes i el valor de la carta de triomf
                    #                         similar_keys = {key for key in self.__policy.keys() if trump_suit == key[0] and played_cards_state == key[1] and hand_cards_state == key[2] and singed_state == key[3] and rules_state == key[5]}
                    #                         # print("choose 2")
                    #
                    #                     if len(similar_keys) == 0:
                    #                         # Es treuen les cartes vistes, les regles i el valor de la carta de triomf
                    #                         similar_keys = {key for key in self.__policy.keys() if trump_suit == key[0] and played_cards_state == key[1] and hand_cards_state == key[2] and singed_state == key[3]}
                    #                         # print("choose 3")
                    #
                    #                     if len(similar_keys) == 0:
                    #                         # Es treuen els cants i el valor de la carta de triomf
                    #                         similar_keys = {key for key in self.__policy.keys() if trump_suit == key[0] and played_cards_state == key[1] and hand_cards_state == key[2] and viewed_cards_state == key[4] and rules_state == key[5]}
                    #                         # print("choose 4")
                    #
                    #                     if len(similar_keys) == 0:
                    #                         # Es treuen els cants, les regles i el valor de la carta de triomf
                    #                         similar_keys = {key for key in self.__policy.keys() if trump_suit == key[0] and played_cards_state == key[1] and hand_cards_state == key[2] and viewed_cards_state == key[4]}
                    #                         # print("choose 5")
                    #
                    #                     if len(similar_keys) == 0:
                    #                         # Es treuen els cants, les cartes vistes i el valor de la carta de triomf
                    #                         similar_keys = {key for key in self.__policy.keys() if trump_suit == key[0] and played_cards_state == key[1] and hand_cards_state == key[2] and rules_state == key[5]}
                    #                         # print("choose 6")

                    if len(similar_keys) == 0:
                        # Es treuen els cants, les cartes vistes, regles i el valor de la carta de triomf
                        similar_keys = {key for key in self.__policy.keys() if trump_suit == key[0] and played_cards_state == key[1] and hand_cards_state == key[2]}
                        # print("choose 7")

                    if len(similar_keys) == 0:
                        print("Random")
                        if player_id == 0:
                            self.__action_p1 = np.random.choice(self.__action_space_p1)
                        elif player_id == 1:
                            self.__action_p2 = np.random.choice(self.__action_space_p2)
                        elif player_id == 2:
                            self.__action_p3 = np.random.choice(self.__action_space_p3)
                        elif player_id == 3:
                            self.__action_p4 = np.random.choice(self.__action_space_p4)
                    else:
                        # Obtenir els valors de les claus
                        values = [self.__policy[key] for key in similar_keys]
                        value_counter = Counter(values)
                        max_value, count_value = value_counter.most_common(1)[0]

                        # Imprimir el resultado
                        print("Action: ", max_value)

                        if player_id == 0:
                            self.__action_p1 = max_value
                        elif player_id == 1:
                            self.__action_p2 = max_value
                        elif player_id == 2:
                            self.__action_p3 = max_value
                        elif player_id == 3:
                            self.__action_p4 = max_value
                else:
                    # Buscarem si hem vist situacions semblants per no donar una acció aleatòria (per exemple, sense cartes vistes)

                    trump_suit = self.__state_p1[0]
                    # trump_card = self.__state_p1[1]
                    played_cards_state = self.__state_p1[1]
                    hand_cards_state = self.__state_p1[2]
                    singed_state = self.__state_p1[3]
                    viewed_cards_state = self.__state_p1[4]
                    rules_state = self.__state_p1[5]

                    # Es treu el valor de la carta de triomf
                    similar_keys = {key for key in self.__policy.keys() if trump_suit == key[0] and played_cards_state == key[1] and hand_cards_state == key[2] and viewed_cards_state == key[4] and rules_state == key[5]}
                    # print("choose 0")

                    #                     if len(similar_keys) == 0:
                    #                         # Es treuen les regles i el valor de la carta de triomf
                    #                         similar_keys = {key for key in self.__policy.keys() if trump_suit == key[0] and played_cards_state == key[1] and hand_cards_state == key[2] and viewed_cards_state == key[4]}
                    #                         print("choose 1")
                    #
                    #                     if len(similar_keys) == 0:
                    #                         # Es treuen les cartes vistes i el valor de la carta de triomf
                    #                         similar_keys = {key for key in self.__policy.keys() if trump_suit == key[0] and played_cards_state == key[1] and hand_cards_state == key[2] and rules_state == key[5]}
                    #                         print("choose 2")

                    if len(similar_keys) == 0:
                        # Es treuen les cartes vistes, les regles i el valor de la carta de triomf
                        similar_keys = {key for key in self.__policy.keys() if trump_suit == key[0] and played_cards_state == key[1] and hand_cards_state == key[2]}
                        print("choose 3")

                    if len(similar_keys) == 0:
                        # print("Random")
                        if player_id == 0:
                            self.__action_p1 = np.random.choice(self.__action_space_p1)
                        elif player_id == 1:
                            self.__action_p2 = np.random.choice(self.__action_space_p2)
                        elif player_id == 2:
                            self.__action_p3 = np.random.choice(self.__action_space_p3)
                        elif player_id == 3:
                            self.__action_p4 = np.random.choice(self.__action_space_p4)
                    else:
                        # Obtenir els valors de les claus
                        values = [self.__policy[key] for key in similar_keys]
                        value_counter = Counter(values)
                        max_value, count_value = value_counter.most_common(1)[0]

                        # Imprimir el resultado
                        # print("Action: ", max_value)
                        if player_id == 0:
                            self.__action_p1 = max_value
                        elif player_id == 1:
                            self.__action_p2 = max_value
                        elif player_id == 2:
                            self.__action_p3 = max_value
                        elif player_id == 3:
                            self.__action_p4 = max_value

        if player_id == 0:
            return self.__action_p1
        elif player_id == 1:
            return self.__action_p2
        elif player_id == 2:
            return self.__action_p3
        elif player_id == 3:
            return self.__action_p4

    def new_episode(self):
        self.__states_actions_returns_p1 = []
        self.__states_actions_returns_p2 = []
        self.__states_actions_returns_p3 = []
        self.__states_actions_returns_p4 = []
        self.__memories_p1 = []
        self.__memories_p2 = []
        self.__memories_p3 = []
        self.__memories_p4 = []

    def set_action_space(self, action_space, player_id: int):
        if player_id == 0:
            self.__action_space_p1 = action_space
            # if self.__state_p1 not in self.__policy_actions:
            #     self.__policy_actions[self.__state_p1] = {}
            #    self.__policy_actions[self.__state_p1]["a"] = action_space
        elif player_id == 1:
            self.__action_space_p2 = action_space
            # if self.__state_p2 not in self.__policy_actions:
            #     self.__policy_actions[self.__state_p2] = {}
            #     self.__policy_actions[self.__state_p2]["a"] = action_space
        elif player_id == 2:
            self.__action_space_p3 = action_space
            # if self.__state_p3 not in self.__policy_actions:
            #     self.__policy_actions[self.__state_p3] = {}
            #     self.__policy_actions[self.__state_p3]["a"] = action_space
        elif player_id == 3:
            self.__action_space_p4 = action_space
            # if self.__state_p4 not in self.__policy_actions:
            #     self.__policy_actions[self.__state_p4] = {}
            #     self.__policy_actions[self.__state_p4]["a"] = action_space

    def __decrease_eps(self):
        # A cada partida que passa, la taxa d'exploració va perdent força
        self.__eps -= self.__eps_decrease if self.__eps - self.__eps_decrease > 0 else 0

    def __calculate_returns(self):
        # Retorn total acumulat al llarg de la partida
        g = 0
        last = True
        state = None

        # Es recorre la memoria a la inversa (del final a l'inici)
        for state, action, reward, available_actions in reversed(self.__memories_p1):
            # L'acció final no s'afegeix a la llista, encara s'ha de calcular
            if last:
                last = False
            else:
                # S'afegeix a la llista que conte els estats, accions i returns al llarg de la partida
                self.__states_actions_returns_p1.append((state, action, g, available_actions))

            g = self.__gamma * g + reward

        if state is not None:
            self.__states_actions_returns_p1.append((state, action, g, available_actions))

        # Es capgira la llista per tenir-la en ordre d'ocurrencia (de principi del joc a final)
        self.__states_actions_returns_p1.reverse()

        # Es recorre la memoria a la inversa (del final a l'inici)
        for state, action, reward, available_actions in reversed(self.__memories_p2):
            # L'acció final no s'afegeix a la llista, encara s'ha de calcular
            if last:
                last = False
            else:
                # S'afegeix a la llista que conte els estats, accions i returns al llarg de la partida
                self.__states_actions_returns_p2.append((state, action, g, available_actions))

            g = self.__gamma * g + reward

        if state is not None:
            self.__states_actions_returns_p2.append((state, action, g, available_actions))

        # Es capgira la llista per tenir-la en ordre d'ocurrencia (de principi del joc a final)
        self.__states_actions_returns_p2.reverse()

        # Es recorre la memoria a la inversa (del final a l'inici)
        for state, action, reward, available_actions in reversed(self.__memories_p3):
            # L'acció final no s'afegeix a la llista, encara s'ha de calcular
            if last:
                last = False
            else:
                # S'afegeix a la llista que conte els estats, accions i returns al llarg de la partida
                self.__states_actions_returns_p3.append((state, action, g, available_actions))

            g = self.__gamma * g + reward

        if state is not None:
            self.__states_actions_returns_p3.append((state, action, g, available_actions))

        # Es capgira la llista per tenir-la en ordre d'ocurrencia (de principi del joc a final)
        self.__states_actions_returns_p3.reverse()

        # Es recorre la memoria a la inversa (del final a l'inici)
        for state, action, reward, available_actions in reversed(self.__memories_p4):
            # L'acció final no s'afegeix a la llista, encara s'ha de calcular
            if last:
                last = False
            else:
                # S'afegeix a la llista que conte els estats, accions i returns al llarg de la partida
                self.__states_actions_returns_p4.append((state, action, g, available_actions))

            g = self.__gamma * g + reward

        if state is not None:
            self.__states_actions_returns_p4.append((state, action, g, available_actions))

        # Es capgira la llista per tenir-la en ordre d'ocurrencia (de principi del joc a final)
        self.__states_actions_returns_p4.reverse()

    def __sar(self, state, action, g, available_actions):
        # Llista per guardar els espais i accions vistes a la partida
        states_actions_visited = []

        sa = (state, action)

        # En principi mai es repeteix. Estem parlant d'una mateixa partida
        if sa not in states_actions_visited:
            # Si existeix l'actualitzo, sino el creo a 1
            if sa in self.__q_pairs_visited:
                self.__q_pairs_visited[sa]["pv"] += 1
            else:
                self.__q_pairs_visited[sa] = {}
                self.__q_pairs_visited[sa]["pv"] = 1

            # incremental implementation -> amb això s'obté la mitjana dels returns per a un estat concret
            #   sense haver de calcular cada cop la mitjana amb tots els results anteriors.
            # https://www.analyticsvidhya.com/blog/2018/11/reinforcement-learning-introduction-monte-carlo-learning-openai-gym/
            # Every visit monte carlo

            # new estimate = 1 / N * [sample - old estimate]
            # Es calcula la nova estimació per aquest parell (estat - acció)
            # if sa in self.__returns:
            if sa in self.__q_pairs_visited and "q" in self.__q_pairs_visited[sa]:
                # self.__returns[sa] += (1 / self.__pairs_visited[sa]) * (g - self.__returns[sa])
                self.__q_pairs_visited[sa]["q"] += (1 / self.__q_pairs_visited[sa]["pv"]) * (
                            g - self.__q_pairs_visited[sa]["q"])
            else:
                # self.__returns[sa] = (1 / self.__pairs_visited[sa]) * g
                self.__q_pairs_visited[sa]["q"] = (1 / self.__q_pairs_visited[sa]["pv"]) * g

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
                    if (state, a) in self.__q_pairs_visited:
                        values.append(self.__q_pairs_visited[(state, a)]["q"])
                values = np.array(values)
                # values = np.array([Q[(state, a)] for a in actionSpace])
                # En cas d'empat, es tria una aleatoria
                # if len(values) > 0:
                best = np.random.choice(np.where(values == values.max())[0])

                # S'actualitza la política amb la millor acció
                # self.__policy[state] = self.__action_space[best]
                self.__policy[state] = available_actions[best]
            else:
                # S'actualitza la política amb una acció aleatoria
                # TODO -> Jo tindre només les accions possibles
                # self.__policy[state] = np.random.choice(self.__action_space)
                self.__policy[state] = np.random.choice(available_actions)

    def update_policy(self):
        self.__calculate_returns()

        # Es recorre la llista d'estats, accions i retorns en ordre de la partida
        for state, action, g, available_actions in self.__states_actions_returns_p1:
            self.__sar(state, action, g, available_actions)

        # Es recorre la llista d'estats, accions i retorns en ordre de la partida
        for state, action, g, available_actions in self.__states_actions_returns_p2:
            self.__sar(state, action, g, available_actions)

        # Es recorre la llista d'estats, accions i retorns en ordre de la partida
        for state, action, g, available_actions in self.__states_actions_returns_p3:
            self.__sar(state, action, g, available_actions)

        # Es recorre la llista d'estats, accions i retorns en ordre de la partida
        for state, action, g, available_actions in self.__states_actions_returns_p4:
            self.__sar(state, action, g, available_actions)

        self.__decrease_eps()
