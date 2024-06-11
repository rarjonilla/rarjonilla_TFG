import copy
from uuid import UUID
import random
from typing import Dict, Optional, List, Tuple
from game_environment.hand import Hand
from game_environment.card import Card
from training.NeuralNetwork import Neural_network
from training.player_state import Player_state
from training.reinforcement.monte_carlo import Monte_carlo
from training.reinforcement.monte_carlo_multiple_key import Monte_carlo_multiple_state


class Player:
    """
    Classe Jugador
    """
    def __init__(self, player_id: int, model_type: int, model_path: Optional[str], rules: Dict, rl_eps: float, rl_eps_decrease: float, rl_gamma: float, rl_agent: Optional[Monte_carlo], training: bool = False) -> None:
        # Id del jugador (0 a 3)
        self.__id: int = player_id
        # Mà del jugador
        self.__hand: Hand = Hand(rules['only_assist'])
        # Indica si s'està fent un entrenament
        self.__training = training

        # Training and IA
        # Tipus i path de l'agent
        self.__model_type: int = model_type
        self.__model_path: str = model_path
        # Regles que s'estan aplicant a la partida
        self.__rules: Dict = rules

        # Es crea la xarxa neuronal (SL o GA) o l'agent per reforç (RL o GA)
        self.nn = None
        self.rl_agent: Monte_carlo = None
        if model_type != 1 and model_type != 9 and model_type != 10:
            # 1 random, 9 i 10 RL
            self.nn = Neural_network(model_type, model_path)
        elif model_type == 9:
            # Es crea l'agent per el jugador o s'utilitza l'agent únic
            if rl_agent is None:
                self.rl_agent = Monte_carlo(rl_eps, rl_eps_decrease, rl_gamma, model_type, model_path)
            else:
                self.rl_agent = rl_agent
        elif model_type == 10:
            # Es crea l'agent per el jugador o s'utilitza l'agent únic
            if rl_agent is None:
                self.rl_agent = Monte_carlo_multiple_state(rl_eps, rl_eps_decrease, rl_gamma, model_type, model_path, self.__training)
            else:
                self.rl_agent = rl_agent

    # Getters
    def get_id(self) -> int:
        return self.__id

    def is_model_type_random(self) -> bool:
        return self.__model_type == 1

    def is_model_type_rl(self) -> bool:
        return self.__model_type == 9 or self.__model_type == 10

    # Functions
    def del_model(self) -> None:
        # Alliberament de memòria RAM en finalitzar les simulacions
        if self.nn is not None:
            self.nn.del_model()
            del self.nn
        elif self.rl_agent is not None:
            self.rl_agent.del_model()
            del self.rl_agent

    def __is_rule_active(self, rule_key: str) -> bool:
        return self.__rules[rule_key]

    def get_next_action(self, there_is_trump_card: bool, change_card_is_higher_than_seven: bool, trump_suit_id: int, highest_suit_card: Optional[Card], deck_has_cards: bool, highest_trump_played: Optional[Card], player_state: Player_state = None) -> Tuple[int, Optional[Card]]:
        # Es tria la següent acció segons el tipus de model
        if self.__model_type != 1 and self.__model_type != 9 and self.__model_type != 10:
            # Xarxa neuronal
            return self.__get_next_action_NN(there_is_trump_card, change_card_is_higher_than_seven, trump_suit_id, highest_suit_card, deck_has_cards, highest_trump_played, player_state)
        elif self.__model_type == 9 or self.__model_type == 10:
            # Agent per reforç
            return self.__get_next_action_RL(there_is_trump_card, change_card_is_higher_than_seven, trump_suit_id, highest_suit_card, deck_has_cards, highest_trump_played, player_state)
        elif self.__model_type == 1:
            # Random
            return self.__get_next_action_random(there_is_trump_card, change_card_is_higher_than_seven, trump_suit_id, highest_suit_card, deck_has_cards, highest_trump_played)
        else:
            raise Exception("No existeix aquest tipus")

    def __get_next_action_random(self, there_is_trump_card: bool, change_card_is_higher_than_seven: bool, trump_suit_id: int, highest_suit_card: Optional[Card], deck_has_cards: bool, highest_trump_played: Optional[Card]) -> Tuple[int, Optional[Card]]:
        # Es calcula les possibles accions del torn i se'n tria una a l'atzar
        if self.__is_rule_active('can_change') and there_is_trump_card and self.__hand.can_change(change_card_is_higher_than_seven, trump_suit_id):
            if random.randint(0, 1):
                return 0, None

        playable_cards: List[int] = self.__hand.get_playable_cards_positions(trump_suit_id, highest_suit_card, deck_has_cards, highest_trump_played)
        chosen_card_position: int = random.randint(0, len(playable_cards) - 1)
        position, card = self.__hand.get_card_in_position(playable_cards[chosen_card_position])

        return position, card

    def __get_next_action_NN(self, there_is_trump_card: bool, change_card_is_higher_than_seven: bool, trump_suit_id: int, highest_suit_card: Optional[Card], deck_has_cards: bool, highest_trump_played: Optional[Card], player_state: Player_state) -> Tuple[int, Optional[Card]]:
        # 1,2,4,5,6,8 -> Supervised Training
        # 3 i 7 -> Genetic
        # TODO -> Eliminar 2 a 6 (només utilitzo la 7 i las 8)

        # Per a les genetiques inicials (random), es donava el cas que tots els outputs de les cartes disponibles a la mà donaven 0, això provocava que es retornes "0" (el valor inicial) com a millor acció, que indica intercanvi (quan realment no pot intercanviar)
        # S'inica -1000 per resoldre aquest error
        best_action: int = -1000
        best_result: int = -1000

        if self.__model_type < 7 or self.__model_type == 8:
            # Supervisada
            # La supervisada avalua cada acció i selecciona l'output amb major probabilitat
            # Els outputs van de menys puntuació a més puntuació, per tant, com més "a la dreta" està l'output, millor és l'acció
            # Si pot fer n accions, s'avalua n vegades i en triarà aquella acció que el seu output està més "a la dreta"
            best_position: int = 0

            if self.__is_rule_active('can_change') and there_is_trump_card and self.__hand.can_change(change_card_is_higher_than_seven, trump_suit_id):
                # El jugador pot intercanviar la carta de triomf
                # S'agafa l'estat del joc per aquest torn (una còpia, ja que es modificarà i només volem emmagatzemar la que acabi sent triada)
                player_state_c = copy.deepcopy(player_state)
                player_state_c.set_action(0, 0)
                inputs_array: List[int] = player_state_c.get_inputs_array()

                if self.__model_type == 3 or self.__model_type == 5:
                    # Cas d'etiqueta Win or Lose (una sola sortida)
                    best_result = self.nn.evaluate_model_one_output(inputs_array)
                    best_action = 0
                elif self.__model_type == 2 or self.__model_type == 4 or self.__model_type == 6 or self.__model_type == 8:
                    # Cas d'etiqueta puntuació i suma de puntuació i heurístic
                    # La supervisada avalua l'acció i selecciona l'output amb major probabilitat i la que més puntuació dona
                    max_position, result = self.nn.evaluate_model_n_outputs(inputs_array)
                    best_position = max_position
                    best_result = result
                    best_action = 0

            # Selecció de les cartes jugables (s'ha d'avaluar per a cadascuna)
            playable_cards_positions: List[int] = self.__hand.get_playable_cards_positions(trump_suit_id, highest_suit_card, deck_has_cards, highest_trump_played)

            for card_pos in playable_cards_positions:
                card_pos, card_in_hand = self.__hand.get_card_in_position_no_remove(card_pos)

                # S'agafa l'estat del joc per aquest torn (una còpia, ja que es modificarà i només volem emmagatzemar la que acabi sent triada)
                player_state_c = copy.deepcopy(player_state)
                player_state_c.set_action(card_in_hand.get_training_idx(), 1)
                inputs_array: List[int] = player_state_c.get_inputs_array()

                if self.__model_type == 3 or self.__model_type == 5:
                    # Cas d'etiqueta Win or Lose (una sola sortida)
                    result = self.nn.evaluate_model_one_output(inputs_array)
                    best_result = max(best_result, result)

                    # Si el resultat d'aquesta acció és millor que la major fins al moment, s'actualitza
                    if best_result == result:
                        best_action = card_pos + 1
                elif self.__model_type == 2 or self.__model_type == 4 or self.__model_type == 6 or self.__model_type == 8:
                    # Cas d'etiqueta puntuació i suma de puntuació i heurístic
                    # La supervisada avalua l'acció i selecciona l'output amb major probabilitat i la que més puntuació dona
                    max_position, result = self.nn.evaluate_model_n_outputs(inputs_array)

                    # Si la posicio és millor, s'ha de canviar, però si es igual a l'anterior s'ha de comprovar el result
                    if max_position > best_position:
                        best_result = result
                        best_action = card_pos + 1
                        best_position = max_position
                    elif best_position == max_position:
                        if result > best_result:
                            best_action = card_pos + 1
                            best_result = result

            # Es retorna la millor acció (es diferencia entre intercanvi o jugar carta)
            if self.__model_type == 3 or self.__model_type == 5:
                if best_action == 0:
                    return 0, None
                else:
                    position, card = self.__hand.get_card_in_position(best_action - 1)
                    return position, card
            elif self.__model_type == 2 or self.__model_type == 4 or self.__model_type == 6 or self.__model_type == 8:
                if best_action == 0:
                    return 0, None
                else:
                    position, card = self.__hand.get_card_in_position(best_action - 1)
                    return position, card

        elif self.__model_type == 7:
            # Genetica
            # La genetica té tants outputs com accions posibles
            # S'avalua un sol cop i es comprova, per a cada posible acció (cartes a la mà o si pot realitzar intercanvi), quina és la que té un valor més alt
            # Posicions 0-39 -> cartes, Posició 40 -> intercanvi
            inputs_array: List[int] = player_state.get_inputs_array()
            results = self.nn.evaluate_model_n_outputs_genetic(inputs_array)

            if self.__is_rule_active('can_change') and there_is_trump_card and self.__hand.can_change(change_card_is_higher_than_seven, trump_suit_id):
                # Acció de pot intercanviar carta
                best_action = 0
                best_result = results[40]

            # Cartes jugables
            playable_cards_positions: List[int] = self.__hand.get_playable_cards_positions(trump_suit_id, highest_suit_card, deck_has_cards, highest_trump_played)
            for card_pos in playable_cards_positions:
                card_pos, card_in_hand = self.__hand.get_card_in_position_no_remove(card_pos)

                # Es comprova el valor del resultat de la posició de la carta
                result = results[card_in_hand.get_training_idx()]

                # Si és millor, s'actualitza
                if result > best_result:
                    best_action = card_pos + 1
                    best_result = result

            # Es retorna la millor acció (es diferencia entre intercanvi o jugar carta)
            if best_action == 0:
                # print("intercanviar carta")
                return 0, None
            else:
                position, card = self.__hand.get_card_in_position(best_action - 1)
                # print("jugar carta", position, card)
                return position, card

    def __get_next_action_RL(self, there_is_trump_card: bool, change_card_is_higher_than_seven: bool, trump_suit_id: int, highest_suit_card: Optional[Card], deck_has_cards: bool, highest_trump_played: Optional[Card], player_state: Player_state) -> Tuple[int, Optional[Card]]:
        # 9 / 10 RL Montecarlo
        # S'agafa l'estat del joc per aquest torn
        inputs_array: List[int] = player_state.get_inputs_array()

        if self.__model_type == 9:
            # RL amb un únic estat
            # Es transforma l'estat al seu valor decimal
            inputs_array_str: str = ''.join(map(str, inputs_array))
            state: int = int(inputs_array_str, 2)
        else:
            # RL amb múltiples estats
            # Es transforma cada part de l'estat al seu valor decimal
            state_list: List[int] = []
            for input_array in inputs_array:
                if len(input_array) > 0:
                    input_array_str: str = ''.join(map(str, input_array))
                    state_list.append(int(input_array_str, 2))
                else:
                    state_list.append(0)

            state = tuple(state_list)

        # s'indica a l'agent l'estat en format decimal
        self.rl_agent.set_state(state, self.__id)

        # Accions que pot dur a terme
        actions: List[int] = []

        if self.__is_rule_active('can_change') and there_is_trump_card and self.__hand.can_change(change_card_is_higher_than_seven, trump_suit_id):
            # Intervanvi de carta
            actions.append(40)

        # Cartes jugables
        playable_cards_positions: List[int] = self.__hand.get_playable_cards_positions(trump_suit_id, highest_suit_card, deck_has_cards, highest_trump_played)

        for card_pos in playable_cards_positions:
            card_pos, card_in_hand = self.__hand.get_card_in_position_no_remove(card_pos)
            actions.append(card_in_hand.get_training_idx())

        # S'actualitza l'espai d'accions disponibles per aquest estat
        self.rl_agent.set_action_space(actions, self.__id)
        # Es tria una acció de la política de l'agent
        action = self.rl_agent.choose_action_from_policy(self.__id)

        # Es retorna la millor acció (es diferencia entre intercanvi o jugar carta)
        if action == 40:
            return 0, None
        else:
            card_pos = actions.index(action)
            card_pos -= 1 if 40 in actions else 0

            position, card = self.__hand.get_card_in_position(card_pos)
            return position, card

    def get_next_action_sing_declarations(self, trump_suit_id: int, player_state: Player_state) -> Optional[int]:
        # Es tria quin cant es fa segons el tipus de l'agent
        if self.__model_type == 2 or self.__model_type == 3 or self.__model_type == 4 or self.__model_type == 6 or self.__model_type == 8:
            return self.__get_next_action_NN_sing_declaration(trump_suit_id, player_state)
        elif self.__model_type == 9 or self.__model_type == 10:
            return self.__get_next_action_RL_sing_declaration(trump_suit_id, player_state)
        elif self.__model_type == 1:
            return self.__get_next_action_random_sing_declaration(trump_suit_id)
        else:
            raise Exception("No existeix tipus NN")

    # Selecció aleatòria
    def __get_next_action_random_sing_declaration(self, trump_suit_id: int) -> Optional[int]:
        # Si el jugador pot declarar més de 1 tute, ha de decidir quin vol triar

        # Es comprova si existeixen tutes a la seva mà
        sing_suits_ids: List[int] = self.__hand.sing_suits_in_hand()

        # Si un d'ells és les 40 no es pot triar
        if trump_suit_id in sing_suits_ids:
            return trump_suit_id
        elif len(sing_suits_ids) > 0:
            # Es tria un d'ells al atzar
            tute_index = random.randint(0, len(sing_suits_ids)) - 1
            return sing_suits_ids[tute_index]
        else:
            return None

    def __get_next_action_NN_sing_declaration(self, trump_suit_id: int, player_state: Player_state) -> Optional[int]:
        # Si el jugador pot declarar més de 1 tute, ha de decidir quin vol triar

        # Es comprova si existeixen tutes a la seva mà
        sing_suits_ids: List[int] = self.__hand.sing_suits_in_hand()
        # Si un d'ells és les 40 no es pot triar
        if trump_suit_id in sing_suits_ids:
            return trump_suit_id
        elif len(sing_suits_ids) > 0:
            # Es tria un d'ells

            # El funcionament és el mateix que per a la tria d'una acció, però ara només es tenen en compte els cants
            best_action: int = -1000
            best_result: int = -1000
            best_position: int = 0

            if self.__model_type < 7:
                for ss in sing_suits_ids:
                    player_state_c = copy.deepcopy(player_state)
                    player_state_c.set_action(ss, 2)
                    inputs_array: List[int] = player_state_c.get_inputs_array()

                    if self.__model_type == 3 or self.__model_type == 5:
                        result = self.nn.evaluate_model_one_output(inputs_array)
                        best_result = max(best_result, result)

                        if best_result == result:
                            best_action = ss
                    elif self.__model_type == 2 or self.__model_type == 4 or self.__model_type == 6 or self.__model_type == 8:
                        max_position, result = self.nn.evaluate_model_n_outputs(inputs_array)

                        if max_position > best_position:
                            best_result = result
                            best_action = ss
                            best_position = max_position
                        elif best_position == max_position:
                            if result > best_result:
                                best_action = ss
                                best_result = result
                return best_action
            elif self.__model_type == 7:
                inputs_array: List[int] = player_state.get_inputs_array()
                results = self.nn.evaluate_model_n_outputs_genetic(inputs_array)

                # Posicions 41-44 -> posibles cants

                # ss és el idx de suit de possibles cants (de 1 a 4), com que les posicions son de la 41 a la 44 a results, hem de sumar-li 40
                for ss in sing_suits_ids:
                    result = results[ss + 40]

                    if result > best_result:
                        best_action = ss
                        best_result = result

                return best_action

        else:
            return None

    def __get_next_action_RL_sing_declaration(self, trump_suit_id: int, player_state: Player_state) -> Tuple[int, Optional[Card]]:
        # 9 / 10 RL Montecarlo
        # El funcionament és el mateix que per a la tria d'una acció, però ara només es tenen en compte els cants

        inputs_array: List[int] = player_state.get_inputs_array()
        if self.__model_type == 9:
            inputs_array_str: str = ''.join(map(str, inputs_array))
            state: int = int(inputs_array_str, 2)
        else:
            state_list: List[int] = []
            for input_array in inputs_array:
                if len(input_array) > 0:
                    input_array_str: str = ''.join(map(str, input_array))
                    state_list.append(int(input_array_str, 2))
                else:
                    state_list.append(0)

            state = tuple(state_list)

        self.rl_agent.set_state(state, self.__id)

        actions: List[int] = []

        # Es comprova si existeixen tutes a la seva mà
        sing_suits_ids: List[int] = self.__hand.sing_suits_in_hand()

        # Si un d'ells és les 40 no es pot triar
        if trump_suit_id in sing_suits_ids:
            return trump_suit_id
        elif len(sing_suits_ids) > 0:
            # Es tria un d'ells
            for ss in sing_suits_ids:
                actions.append(ss + 40)

            self.rl_agent.set_action_space(actions, self.__id)
            action = self.rl_agent.choose_action_from_policy(self.__id)

            return action - 40

    def init_hand(self) -> None:
        self.__hand = Hand(self.__rules['only_assist'])

    # Hand Functions
    def hand_add_card_to_hand(self, card: Card) -> None:
        self.__hand.add_card(card)

    def hand_add_singed_tute_suit(self, suit_id: int) -> None:
        self.__hand.add_singed_tute_suit(suit_id)

    def hand_black_hand_cards_position(self, trump_suit_id: int) -> Optional[List[int]]:
        return self.__hand.black_hand_cards_position(trump_suit_id)

    def hand_can_change(self, change_card_is_higher_than_seven: bool, round_suit_id: int) -> bool:
        return self.__hand.can_change(change_card_is_higher_than_seven, round_suit_id)

    def hand_card_to_change(self, change_card_is_higher_than_seven: bool, round_suit_id: int) -> Tuple[int, Optional[Card]]:
        return self.__hand.card_to_change(change_card_is_higher_than_seven, round_suit_id)

    def hand_cards_in_hand(self) -> int:
        return self.__hand.cards_in_hand()

    def hand_change_card(self, card_position: int, new_card: Card) -> None:
        self.__hand.change_card(new_card, card_position)

    def hand_get_card_in_position(self, card_position: int) -> Tuple[int, Card]:
        return self.__hand.get_card_in_position(card_position)

    def hand_get_card_in_position_no_remove(self, card_position: int) -> Tuple[int, Card]:
        return self.__hand.get_card_in_position_no_remove(card_position)

    def hand_get_cards_copy(self) -> List[Card]:
        return self.__hand.get_cards_copy()

    def hand_get_playable_cards_positions(self, trump_suit_id: int, highest_suit_card: Optional[Card], deck_has_cards: bool, highest_trump_played: Optional[Card]) -> List[int]:
        return self.__hand.get_playable_cards_positions(trump_suit_id, highest_suit_card, deck_has_cards, highest_trump_played)

    def hand_get_playable_cards(self, trump_suit_id: int, highest_suit_card: Optional[Card], deck_has_cards: bool, highest_trump_played: Optional[Card]) -> List[Card]:
        return self.__hand.get_playable_cards(trump_suit_id, highest_suit_card, deck_has_cards, highest_trump_played)

    def hand_get_sing_cards_position(self, singed_suit_id: int) -> List[int]:
        return self.__hand.get_sing_cards_position(singed_suit_id)

    def hand_get_singed_suits(self) -> List[int]:
        return self.__hand.get_singed_suits()

    def hand_has_black_hand(self, trump_suit_id: int) -> bool:
        return self.__hand.has_black_hand(trump_suit_id)

    def hand_has_cards(self) -> bool:
        return self.hand_cards_in_hand() != 0

    def hand_has_one_left_card(self) -> bool:
        return self.hand_cards_in_hand() == 1

    def hand_has_tute(self) -> Tuple[bool, bool]:
        return self.__hand.has_tute()

    def hand_sing_suits_in_hand(self) -> List[int]:
        return self.__hand.sing_suits_in_hand()

    def hand_singed_tute_suits(self) -> List[int]:
        return self.__hand.get_singed_suits()

    def hand_tute_cards_position(self) -> Optional[List[int]]:
        return self.__hand.tute_cards_position()

    # RL Functions
    def rl_add_memory(self):
        self.rl_agent.add_memory(self.__id)

    def rl_new_episode(self):
        self.rl_agent.new_episode()

    def rl_save_model(self):
        self.rl_agent.save_model()

    def rl_set_reward(self, reward: int):
        self.rl_agent.set_reward(reward, self.__id)

    def rl_update_policy(self):
        self.rl_agent.update_policy()

    # Print
    def show_hand(self) -> None:
        print("Player ", self.__id, end="\r")
        print(self.__hand)
