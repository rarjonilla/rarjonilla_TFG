import copy
from uuid import UUID
import random
from typing import Dict, Optional, List, Tuple
from game_environment.hand import Hand
from game_environment.card import Card
from training.NeuralNetwork import Neural_network
from training.player_state import Player_state


class Player:
    """
    Classe Player...
    """
    def __init__(self, player_id: int, model_type: int, model_path: Optional[str], rules: Dict) -> None:
        self.__id: int = player_id
        self.__hand: Hand = Hand(rules['only_assist'])

        # Training and IA
        self.__model_type: int = model_type
        self.__model_path: str = model_path
        self.__rules: Dict = rules

        # Es crea la xarxa neuronal
        if model_type != 1:
            self.nn = Neural_network(model_type, model_path)

        # info de round
        # self.__actual_round_uuid: Optional[UUID] = None
        #        self.round_step: int = 0

    # Getters
    def get_id(self) -> int:
        return self.__id

    def is_model_type_random(self) -> bool:
        return self.__model_type == 1

    # Functions
    def __is_rule_active(self, rule_key: str) -> bool:
        return self.__rules[rule_key]

    def get_next_action(self, there_is_trump_card: bool, change_card_is_higher_than_seven: bool, trump_suit_id: int, highest_suit_card: Optional[Card], deck_has_cards: bool, highest_trump_played: Optional[Card], player_state: Player_state = None) -> Tuple[int, Optional[Card]]:
        if self.__model_type != 1:
            return self.__get_next_action_NN(there_is_trump_card, change_card_is_higher_than_seven, trump_suit_id, highest_suit_card, deck_has_cards, highest_trump_played, player_state)
        elif self.__model_type == 1:
            return self.__get_next_action_random(there_is_trump_card, change_card_is_higher_than_seven, trump_suit_id, highest_suit_card, deck_has_cards, highest_trump_played)
        else:
            raise Exception("No existeix tipus NN")

    def __get_next_action_random(self, there_is_trump_card: bool, change_card_is_higher_than_seven: bool, trump_suit_id: int, highest_suit_card: Optional[Card], deck_has_cards: bool, highest_trump_played: Optional[Card]) -> Tuple[int, Optional[Card]]:
        """docstring"""
        if self.__is_rule_active('can_change') and there_is_trump_card and self.__hand.can_change(change_card_is_higher_than_seven, trump_suit_id):
            if random.randint(0, 1):
                return 0, None

        playable_cards: List[int] = self.__hand.get_playable_cards_positions(trump_suit_id, highest_suit_card, deck_has_cards, highest_trump_played)
        chosen_card_position: int = random.randint(0, len(playable_cards) - 1)
        position, card = self.__hand.get_card_in_position(playable_cards[chosen_card_position])

        return position, card

        # generar SQL string per guardar la jugada
        # self._generate_sql_string(chosen_action, self.round_step, game_type)
        # self.round_step += 1

    def __get_next_action_NN(self, there_is_trump_card: bool, change_card_is_higher_than_seven: bool, trump_suit_id: int, highest_suit_card: Optional[Card], deck_has_cards: bool, highest_trump_played: Optional[Card], player_state: Player_state) -> Tuple[int, Optional[Card]]:
        best_action: int = 0
        best_result: int = 0
        best_position: int = 0

        if self.__is_rule_active('can_change') and there_is_trump_card and self.__hand.can_change(change_card_is_higher_than_seven, trump_suit_id):
            player_state_c = copy.deepcopy(player_state)
            player_state_c.set_action(0, 0)
            inputs_array: List[int] = player_state_c.get_inputs_array()

            if self.__model_type == 3 or self.__model_type == 5:
                best_result = self.nn.evaluate_model_one_output(inputs_array)
            elif self.__model_type == 2 or self.__model_type == 4 or self.__model_type == 6:
                max_position, result = self.nn.evaluate_model_n_outputs(inputs_array)
                best_position = max_position
                best_result = result

                # print(best_position, best_result)

        playable_cards_positions: List[int] = self.__hand.get_playable_cards_positions(trump_suit_id, highest_suit_card, deck_has_cards, highest_trump_played)

        for card_pos in playable_cards_positions:
            card_pos, card_in_hand = self.__hand.get_card_in_position_no_remove(card_pos)

            player_state_c = copy.deepcopy(player_state)
            player_state_c.set_action(card_in_hand.get_training_idx(), 1)
            inputs_array: List[int] = player_state_c.get_inputs_array()
            if self.__model_type == 3 or self.__model_type == 5:
                result = self.nn.evaluate_model_one_output(inputs_array)
                best_result = max(best_result, result)
                # print(best_result, result)

                if best_result == result:
                    best_action = card_pos + 1
            elif self.__model_type == 2 or self.__model_type == 4 or self.__model_type == 6:
                max_position, result = self.nn.evaluate_model_n_outputs(inputs_array)
                # print(max_position, result)
                # print(best_result, result)
                # print(best_action)
                # print("")

                #print(max_position, result)

                # Si la posicio és millor, s'ha de canviar, però si es igual a l'anterior s'ha de comprovar el result

                # best_position = max(best_position, max_position)

                if max_position > best_position:
                    best_result = result
                    best_action = card_pos + 1
                    best_position = max_position
                elif best_position == max_position:
                    if result > best_result:
                        best_action = card_pos + 1
                        best_result = result

                # best_position = max(best_position, max_position)
                #
                #                 if best_position == max_position:
                #                     best_action = card_pos + 1
                #                     best_result = result

                # print(best_position, best_result, best_action)
                # print("")

        if self.__model_type == 3 or self.__model_type == 5:
            if best_action == 0:
                return 0, None
            else:
                position, card = self.__hand.get_card_in_position(best_action - 1)
                return position, card
        elif self.__model_type == 2 or self.__model_type == 4 or self.__model_type == 6:
            # print("best", best_position, best_result, best_action)
            if best_action == 0:
                return 0, None
            else:
                position, card = self.__hand.get_card_in_position(best_action - 1)
                return position, card

    def get_next_action_sing_declarations(self, trump_suit_id: int, player_state: Player_state) -> Optional[int]:
        if self.__model_type == 2 or self.__model_type == 3 or self.__model_type == 4 or self.__model_type == 6:
            return self.__get_next_action_NN_sing_declaration(trump_suit_id, player_state)
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
            best_action: int = 0
            best_result: int = 0
            best_position: int = 0

            for ss in sing_suits_ids:
                player_state_c = copy.deepcopy(player_state)
                player_state_c.set_action(ss, 2)
                inputs_array: List[int] = player_state_c.get_inputs_array()

                if self.__model_type == 3 or self.__model_type == 5:
                    result = self.nn.evaluate_model_one_output(inputs_array)
                    best_result = max(best_result, result)
                    # print(best_result, result)

                    if best_result == result:
                        best_action = ss
                elif self.__model_type == 2 or self.__model_type == 4 or self.__model_type == 6:
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
        else:
            return None

    def init_hand(self, round_uuid: UUID) -> None:
        self.__hand = Hand(self.__rules['only_assist'])
        self.__actual_round_uuid = round_uuid
        # self.round_step = 0

    def reset_seen_cards(self) -> None:
        pass

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

    # Print
    def show_hand(self) -> None:
        print("Player ", self.__id, end="\r")
        print(self.__hand)
