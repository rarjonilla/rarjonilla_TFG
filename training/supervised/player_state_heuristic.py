import copy
from typing import List, Optional, Dict

from configuration import PRINT_CONSOLE
from game_environment.card import Card

import numpy as np


class Player_state_heuristic:
    """Classe player state heurística"""

    def __init__(self, played_card: Card, playable_hand: List[Card], round_played_cards: List[Optional[Card]], player_id: int, num_players: int, trump: Card, all_played_cards: List[int], single_mode: bool, sing_declarations: List[int], is_brisca: bool) -> None:
        # Cartes jugades a la ronda
        self.__round_played_cards: List[Optional[Card]] = round_played_cards

        self.__is_brisca = is_brisca

        # Cants de la ronda
        self.__sing_declarations: List[int] = sing_declarations
        # Totes les cartes jugades fins al moment
        self.__all_played_cards: List[int] = all_played_cards

        self.__single_mode: bool = single_mode

        # Carta de triomf
        self.__trump: Card = trump

        # Id del jugador del torn actual
        self.__player_id: int = player_id

        # Total de jugadors
        self.__num_players: int = num_players

        # Carta jugada
        self.__played_card = played_card
        # Cartes jugables de la mà
        self.__playable_hand = playable_hand

        # Afegeixo la carta jugada a playable_hand per fer les comprovacions pertinents
        self.__playable_hand.append(self.__played_card)

        # Indica si és el primer torn de la ronda
        self.__is_first_turn: bool = False

        # Indica la carta inicial de la ronda
        self.__initial_card: Optional[Card] = None
        # Indica el número de la carta jugada del pal de la carta inicial més alta de la ronda
        self.__highest_initial_suit_label: int = 0
        # Indica el valor de la carta jugada del pal de la carta inicial més alta de la ronda
        self.__highest_initial_suit_value: int = 0
        # Indica si la carta inicial és del coll del triomf
        self.__is_initial_suit_trump: bool = False
        # Indica si s'ha jugat alguna carta de triomf en aquesta ronda
        self.__trump_used: bool = False
        # Indica el número de la carta jugada del pal del coll més alta de la ronda
        self.__highest_trump_label: int = 0
        # Indica el valor de la carta jugada del pal del coll més alta de la ronda
        self.__highest_trump_value: int = 0
        # La carta de triomf jugada del pal del coll més alta de la ronda
        self.__highest_trump: Card = None
        # Indica si s'ha jugat alguna carta d'alt valor a la ronda
        self.__valuable_cards_played: bool = False
        # Indica si es guanya la ronda (momentàniament) amb la carta triada (poden quedar més jugadors per jugar a la ronda)
        self.__wins_round: bool = False
        # Indica si el company està guanyant la ronda (per equips)
        self.__teammate_wins_round: bool = False
        # Indica el id del company (per equips)
        self.__teammate_id: int = 0
        # Indica la posició de la carta jugada pel company (per equips)
        self.__teammate_played_position: int = 0
        # Indica la la carta jugada pel company (per equips)
        self.__teammate_played_card: Card = None
        # Llista amb el total de cartes altes (As i 3) que s'han jugat durant tota la partida per a cada coll
        self.__high_cards_played: List[int] = []
        # Indica quin és el coll en el qual s'han jugat més cartes altes (As i 3) durant tota la partida
        self.__max_played_as_and_three_suit: int = 0
        # Indica si la carta jugada és del mateix coll que el cant que ha fet el jugador (Tute)
        self.__is_self_sing_suit: bool = False
        # Indica si la carta jugada és del mateix coll que el cant que ha fet el company (Tute per equips)
        self.__is_team_sing_suit: bool = False
        # Indica si el jugador pot jugar una carta del coll de cant d'un rival (Tute)
        self.__has_other_sing_suit: bool = False
        # Indica si la carta jugada és del coll de cant d'un rival (Tute)
        self.__is_other_sing_suit: bool = False
        # Indica si el jugador pot jugar una carta diferent del rei i el cavall del coll del qual pot cantar (Tute)
        self.__can_use_other_than_suit_king_knight: bool = False
        # Indica el coll de cant del jugador (Tute)
        self.__self_singed_suit: int = None
        # Indica el coll de cant del company (Tute per equips)
        self.__team_singed_suit: int = None
        # Indica si el jugador pot guanyar la ronda amb una carta del coll del mateix coll que la carta inicial
        self.__can_win_with_suit: bool = False
        # Indica si el jugador pot guanyar la ronda amb una de valor alt del mateix coll que la carta inicial
        self.__can_win_with_high: bool = False
        # Indica si el jugador pot guanyar la ronda amb una carta del coll del triomf
        self.__can_win_with_trump: bool = False
        # Indica si el jugador té l'As del coll del triomf
        self.__has_trump_as: bool = False
        # Indica si el jugador té cartes de menys valor que la carta jugada
        self.__has_cards_with_less_value: bool = False
        # Indica si el jugador té cartes del coll en el qual s'han jugat més cartes altes (As i 3) durant tota la partida
        self.__has_cards_of_most_played_as_and_three_suit: bool = False
        # Indica si el jugador té cartes de menys valor que la carta de triomf jugada
        self.__has_low_cards_not_trump_suit: bool = False
        # Indica si el jugador té cartes de menys valor o igual que no siguin del coll del triomf
        self.__has_low_or_same_value_not_trump: bool = False
        # Indica si el jugador té cartes de més valor del mateix coll que la carta jugada
        self.__has_more_valuable_card: bool = False
        # Carta de més valor del mateix coll que la carta jugada
        self.__more_valuable_card: Card = None
        # Indica si el jugador disposa d'una carta de triomf de menor valor que la carta jugada que li permet guanyar la ronda
        self.__has_lower_cards_trump_suit_to_win_round: bool = False
        # Carta de triomf de menor valor que la carta jugada que li permet guanyar la ronda
        self.__lower_card_trump_suit_to_win_round: Card = None
        # Indica si el jugador és l'últim de jugar en aquesta ronda
        self.__is_last_round_player: bool = False

        self.__heuristics_calc_variables()

    def __heuristics_calc_variables(self):
        # Calcul de les diferents possibilitats de les cartes jugades que ajudaran al càlcul d'heuristics final
        self.__is_first_turn = True if len(self.__round_played_cards) == 0 else False

        self.__initial_card = None if self.__is_first_turn else self.__round_played_cards[0]

        self.__highest_initial_suit_label = 0 if self.__is_first_turn else self.__initial_card.get_label()
        self.__highest_initial_suit_value = 0 if self.__is_first_turn else self.__initial_card.get_value()

        self.__is_initial_suit_trump = True if not self.__is_first_turn and self.__trump.is_same_suit(self.__initial_card.get_suit_id()) else False
        self.__trump_used = self.__is_initial_suit_trump

        self.__highest_trump_label = 0 if not self.__is_initial_suit_trump else self.__highest_initial_suit_label
        self.__highest_trump_value = 0 if not self.__is_initial_suit_trump else self.__highest_initial_suit_value
        self.__highest_trump = None if not self.__is_initial_suit_trump else self.__initial_card

        self.__teammate_id = (self.__player_id + 2) % self.__num_players
        self.__teammate_played_position = -1

        if not self.__single_mode and len(self.__round_played_cards) == 2:
            self.__teammate_played_position = 0
        elif not self.__single_mode and len(self.__round_played_cards) == 3:
            self.__teammate_played_position = 1

        # Cartes ja jugades
        for played_c in self.__round_played_cards:
            if self.__trump.is_same_suit(played_c.get_suit_id()):
                self.__trump_used = True
                if played_c.has_higher_value(self.__highest_trump_value) or (played_c.has_same_value(self.__highest_trump_value) and played_c.is_higher_label(self.__highest_trump_label)):
                    self.__highest_trump_label = played_c.get_label()
                    self.__highest_trump_value = played_c.get_value()
                    self.__highest_trump = played_c

            if played_c.is_same_suit(self.__initial_card.get_suit_id()) and played_c.has_higher_value(self.__highest_initial_suit_value) or (played_c.has_same_value(self.__highest_initial_suit_value) and played_c.is_higher_label(self.__highest_initial_suit_label)):
                self.__highest_initial_suit_label = played_c.get_label()
                self.__highest_initial_suit_value = played_c.get_value()

            if played_c.is_as() or played_c.is_three():
                self.__valuable_cards_played = True

        if not self.__is_first_turn and not self.__trump_used and self.__played_card.is_same_suit(self.__initial_card.get_suit_id()) and (self.__played_card.has_higher_value(self.__highest_initial_suit_value) or (self.__played_card.has_same_value(self.__highest_initial_suit_value) and self.__played_card.is_higher_label(self.__highest_initial_suit_label))):
            self.__wins_round = True
        elif not self.__is_first_turn and not self.__trump_used and self.__played_card.is_same_suit(self.__trump.get_suit_id()) and (self.__played_card.has_higher_value(self.__highest_trump_value) or (self.__played_card.has_same_value(self.__highest_trump_value) and self.__played_card.is_higher_label(self.__highest_trump_label))):
            self.__wins_round = True
        elif not self.__is_first_turn and self.__trump_used and self.__played_card.is_same_suit(self.__trump.get_suit_id()) and (self.__played_card.has_higher_value(self.__highest_trump_value) or (self.__played_card.has_same_value(self.__highest_trump_value) and self.__played_card.is_higher_label(self.__highest_trump_label))):
            self.__wins_round = True

        if self.__teammate_played_position != -1:
            self.__teammate_played_card = self.__round_played_cards[self.__teammate_played_position]

            if not self.__trump_used and self.__teammate_played_card.is_same_suit(self.__initial_card.get_suit_id()) and self.__teammate_played_card.has_same_value(self.__highest_initial_suit_value) and self.__teammate_played_card.get_label() == self.__highest_initial_suit_label:
                self.__teammate_wins_round = True
            elif not self.__trump_used and self.__teammate_played_card.is_same_suit(self.__trump.get_suit_id()) and self.__teammate_played_card.has_same_value(self.__highest_trump_value) and self.__teammate_played_card.get_label() == self.__highest_trump_label:
                self.__teammate_wins_round = True
            elif self.__trump_used and self.__teammate_played_card.is_same_suit(self.__trump.get_suit_id()) and self.__teammate_played_card.has_same_value(self.__highest_trump_value) and self.__teammate_played_card.get_label() == self.__highest_trump_label:
                self.__teammate_wins_round = True

        # As i 3 jugats de cada coll
        self.__high_cards_played = []
        for i in range(0, 4):
            total_high_played = 0
            position_as = 0 + 10 * i
            position_three = 2 + 10 * i

            total_high_played += 1 if self.__all_played_cards[position_as] == 1 else 0
            total_high_played += 1 if self.__all_played_cards[position_three] == 1 else 0

            self.__high_cards_played.append(total_high_played)

        self.__max_played_as_and_three_suit = max(self.__high_cards_played)

        if self.__sing_declarations is not None:
            self.__self_singed_suit = self.__sing_declarations[self.__player_id]
            if not self.__single_mode and self.__teammate_played_position != -1:
                self.__team_singed_suit = self.__sing_declarations[self.__teammate_played_position]

            if self.__self_singed_suit is not None and self.__played_card.is_same_suit(self.__self_singed_suit):
                self.__is_self_sing_suit = True

            if self.__team_singed_suit is not None and self.__played_card.is_same_suit(self.__team_singed_suit):
                self.__is_team_sing_suit = True

            for i in range(0, self.__num_players):
                if i != self.__player_id:
                    # print(self.__sing_declarations)
                    singed_suit = self.__sing_declarations[i]
                    if singed_suit is not None and self.__played_card.is_same_suit(singed_suit) and not self.__is_self_sing_suit and not self.__is_team_sing_suit:
                        self.__is_other_sing_suit = True

        # Cartes a la mà del jugador
        for card_in_hand in self.__playable_hand:
            if self.__played_card.is_same_suit(self.__trump.get_suit_id()) and card_in_hand.is_same_suit(self.__trump.get_suit_id()) and self.__played_card.has_higher_value(card_in_hand.get_value()):
                if not self.__is_first_turn and not self.__trump_used and card_in_hand.is_same_suit(self.__initial_card.get_suit_id()) and (
                        card_in_hand.has_higher_value(self.__highest_initial_suit_value) or (
                        card_in_hand.has_same_value(self.__highest_initial_suit_value) and card_in_hand.is_higher_label(
                        self.__highest_initial_suit_label))):

                    self.__has_lower_cards_trump_suit_to_win_round = True
                    if self.__lower_card_trump_suit_to_win_round is None or self.__lower_card_trump_suit_to_win_round.has_higher_value(card_in_hand.get_value()):
                        self.__lower_card_trump_suit_to_win_round = card_in_hand
                elif not self.__is_first_turn and not self.__trump_used and card_in_hand.is_same_suit(
                        self.__trump.get_suit_id()) and (self.__played_card.has_higher_value(self.__highest_trump_value) or (
                        card_in_hand.has_same_value(self.__highest_trump_value) and card_in_hand.is_higher_label(
                        self.__highest_trump_label))):

                    self.__has_lower_cards_trump_suit_to_win_round = True
                    if self.__lower_card_trump_suit_to_win_round is None or self.__lower_card_trump_suit_to_win_round.has_higher_value(card_in_hand.get_value()):
                        self.__lower_card_trump_suit_to_win_round = card_in_hand
                elif not self.__is_first_turn and self.__trump_used and card_in_hand.is_same_suit(self.__trump.get_suit_id()) and (
                        card_in_hand.has_higher_value(self.__highest_trump_value) or (
                        card_in_hand.has_same_value(self.__highest_trump_value) and card_in_hand.is_higher_label(
                        self.__highest_trump_label))):

                    self.__has_lower_cards_trump_suit_to_win_round = True
                    if self.__lower_card_trump_suit_to_win_round is None or self.__lower_card_trump_suit_to_win_round.has_higher_value(card_in_hand.get_value()):
                        self.__lower_card_trump_suit_to_win_round = card_in_hand

            if not self.__is_first_turn:
                if not self.__trump_used and card_in_hand.is_same_suit(self.__initial_card.get_suit_id()) and (card_in_hand.is_as() or card_in_hand.is_three()) and (card_in_hand.has_higher_value(self.__highest_initial_suit_value) or (card_in_hand.has_same_value(self.__highest_initial_suit_value) and card_in_hand.is_higher_label(self.__highest_initial_suit_label))):
                    self.__can_win_with_high = True

                if not self.__trump_used and card_in_hand.is_same_suit(self.__initial_card.get_suit_id()) and (card_in_hand.has_higher_value(self.__highest_initial_suit_value) or (card_in_hand.has_same_value(self.__highest_initial_suit_value) and card_in_hand.is_higher_label(self.__highest_initial_suit_label))):
                    self.__can_win_with_suit = True

            if card_in_hand.is_same_suit(self.__trump.get_suit_id()) and (card_in_hand.has_higher_value(self.__highest_trump_value) or (card_in_hand.has_same_value(self.__highest_trump_value) and card_in_hand.is_higher_label(self.__highest_trump_label))):
                self.__can_win_with_trump = True

            if card_in_hand.is_same_suit(self.__trump.get_suit_id()) and card_in_hand.is_as():
                self.__has_trump_as = True

            if self.__played_card.has_higher_value(card_in_hand.get_value()):
                self.__has_cards_with_less_value = True

            # print(not card_in_hand.is_same_suit(self.__trump.get_suit_id()), high_cards_played[card_in_hand.get_suit_id() - 1], max_played_as_and_three_suit)
            if not card_in_hand.is_same_suit(self.__trump.get_suit_id()) and self.__high_cards_played[card_in_hand.get_suit_id() - 1] == self.__max_played_as_and_three_suit:
                self.__has_cards_of_most_played_as_and_three_suit = True

            if not card_in_hand.is_same_suit(self.__trump.get_suit_id()) and card_in_hand.get_value() < 4:
                self.__has_low_cards_not_trump_suit = True

            if self.__played_card.is_same_suit(self.__trump.get_suit_id()) and not card_in_hand.is_same_suit(self.__trump.get_suit_id()) and (self.__played_card.has_higher_value(card_in_hand.get_value()) or self.__played_card.has_same_value(card_in_hand.get_value())):
                self.__has_low_or_same_value_not_trump = True

            if not card_in_hand.is_same_suit(self.__trump.get_suit_id()) and card_in_hand.has_higher_value(self.__played_card.get_value()):
                self.__has_more_valuable_card = True
                if self.__more_valuable_card is None or card_in_hand.has_higher_value(self.__more_valuable_card.get_value()):
                    self.__more_valuable_card = card_in_hand

            if self.__sing_declarations is not None:
                for i in range(0, self.__num_players):
                    if i != self.__player_id:
                        singed_suit = self.__sing_declarations[i]

                        if singed_suit is not None and card_in_hand.is_same_suit(singed_suit):
                            self.__has_other_sing_suit = True

            if self.__self_singed_suit is not None and not self.__played_card.is_king() and not self.__played_card.is_knight() and self.__played_card.is_same_suit(self.__self_singed_suit):
                self.__can_use_other_than_suit_king_knight = True

        self.__is_last_round_player = len(self.__round_played_cards) == self.__num_players - 1

    # La idea és sumar aquest valor heuristic al resultat final de la partida (guanyar la partida seran 30 punts extra a l'heuristica de cada ronda)
    # D'aquesta manera, tot i no guanyar la partida s'haurà qualificat la jugada
    # Explicar que per "round points" no funciona bé (juga cartes molt altes sempre per la aleatorietat de les dades inicials (com que la màquina no juga bé, es donen molts casos on aquesta estrategia fa sumar molts punts, però en una partida real això no funciona))
    # Explicar que per "win or lose" no funciona bé (les jugades que porten a guanyar poden ser molt dolentes per la aleatorietat de les dades inicials (com que la màquina no juga bé, es donen molts casos on jugades dolentes donen lloc a guanyar partides, però en una partida real això no funciona))
    # si l'heuristica funciona, es podrien generar dades aleatories amb aquest model per tal d'entrenar-ne un de nou únicament per "round points" i "win or lose"
    # Aquest primer model amb l'heuristica representaria un conjunt de dades de jugadors semiprofessionals
    def __heuristics_brisca_2j(self):
        # Càlcul de l'heurístic per a la Brisca per a dos jugadors
        local_print_console = False

        heuristic: int = 0

        if self.__is_first_turn:
            if self.__is_brisca and (self.__played_card.is_as() or self.__played_card.is_three()) and self.__has_cards_with_less_value:
                # Iniciar el torn amb As o 3 no és correcte (de vegades és bo depenent si sospites que el rival no te cartes de triomf, però, en general, és una jugada horrible)
                # Restem el valor de la carta per 2
                heuristic -= self.__played_card.get_value() * 2
                if local_print_console:
                    print("heuristic 1", heuristic)

            # Iniciar el torn amb carta del coll del triomf, tenint altres cartes de valor igual o més baix d'altres colls no és correcte
            # Restarem 10 punts
            if self.__played_card.is_same_suit(self.__trump.get_suit_id()) and self.__has_low_or_same_value_not_trump:
                heuristic += -10 if (self.__is_brisca or (not self.__is_brisca and self.__is_other_sing_suit)) else 0
                if local_print_console:
                    print("heuristic 2", heuristic)

            # S'ha de prioritzar utilitzar cartes de colls on ja s'hagin jugat el 3 i l'As (que no siguin triomf), si s'en disposa
            # Es sumaran / restaran 5 punts
            t: int = self.__high_cards_played[self.__played_card.get_suit_id() - 1]
            if self.__max_played_as_and_three_suit > 0:
                if self.__has_cards_of_most_played_as_and_three_suit and not t == self.__max_played_as_and_three_suit:
                    heuristic -= 5
                elif t == self.__max_played_as_and_three_suit and not self.__played_card.is_same_suit(self.__trump.get_suit_id()):
                    # Sumem només si no es tracta del coll del triomf
                    heuristic += 5

                if local_print_console:
                    print("heuristic 3", heuristic)

            # S'esta utilitzant una carta del coll del qual el propi jugador o el seu company han cantat
            # Error molt greu, restem 20 punts
            if not self.__is_brisca and (self.__is_self_sing_suit or self.__is_team_sing_suit):
                heuristic -= 20

            if not self.__is_brisca and self.__is_other_sing_suit:
                # S'esta utilitzant una carta del coll del qual un rival ha cantat
                # Bona jugada, sumem 20 punts
                heuristic += 20
            elif not self.__is_brisca and not self.__is_other_sing_suit and self.__has_other_sing_suit:
                # S'esta utilitzant una carta que no és del coll cantat pel rival
                # Error molt greu, restem 20 punts (només si es disposa de carta del coll cantat)
                heuristic -= 20

            if local_print_console:
                print("heuristic 4", heuristic)

        else:
            # Final de torn o entremig

            # Si s'ha jugat el 3 del coll del triomf i el jugador té l'As, jugar-lo és correcte, no jugar-lo incorrecte
            # Sumem / restem 5 punts
            if self.__highest_trump is not None and self.__highest_trump.is_three() and self.__played_card.is_same_suit(self.__trump.get_suit_id()) and self.__played_card.is_as():
                heuristic += 5
            elif self.__highest_trump is not None and self.__highest_trump.is_three() and self.__has_trump_as:
                heuristic -= 5

            if local_print_console:
                print("heuristic 1", heuristic)

            if len(self.__round_played_cards) == self.__num_players - 1 and not self.__single_mode and self.__is_last_round_player:
                if self.__teammate_wins_round and self.__has_more_valuable_card:
                    # Si el company guanya la ronda, s'ha de jugar la carta amb més valor de la mà que no sigui triomf
                    # Sempre que no falti algun jugador rival per tirar
                    # En aquest cas està jugant una carta amb menys valor de les disponibles
                    # Restem la diferència del valor de les cartes (no és el mateix jugar un 3 tenint un As on es perd 1 punt que jugar un 2 tenint un As on es perden 11 punts)
                    heuristic -= self.__more_valuable_card.get_value() - self.__played_card.get_value()
                elif self.__teammate_wins_round and not self.__has_more_valuable_card:
                    # Si juga la carta amb més valor, afegirem aquests punts extra per tal de bonificar encara més aquesta jugada
                    heuristic += self.__played_card.get_value()

                if local_print_console:
                    print("heuristic 2", heuristic)

            if not self.__wins_round and self.__valuable_cards_played:
                # No guanya ronda on s'han jugat cartes valuoses quan podria haver-ho fet
                if not self.__teammate_wins_round:
                    if self.__can_win_with_high:
                        # Podia guanyar la ronda amb un As o 3, restem 10
                        heuristic -= 10
                    elif self.__can_win_with_trump:
                        # Podia guanyar la ronda amb qualsevol carta de triomf, restem 10
                        heuristic -= 10
                    elif self.__can_win_with_suit:
                        # Podia guanyar la ronda amb qualsevol carta del col de ronda, restem 4
                        heuristic -= 4
                    if local_print_console:
                        print("heuristic 3 if", heuristic)
                else:
                    # Cas on el company està guanyant la ronda
                    if self.__can_win_with_high:
                        # Podíem guanyar la ronda amb un As o 3 del coll de ronda, restem 10 punts
                        heuristic -= 10
                    if local_print_console:
                        print("heuristic 3 else", heuristic)
            elif not self.__wins_round and self.__can_win_with_high:
                if self.__can_win_with_high:
                    # No guanya ronda on no s'han jugat cartes valuoses, però té As o 3 del coll d'inici
                    # Restem 10 punts
                    heuristic -= 10
                if local_print_console:
                    print("heuristic 4", heuristic)
            elif self.__wins_round and self.__valuable_cards_played:
                # Guanya ronda on s'han jugat cartes valuoses
                if self.__can_win_with_high and (self.__played_card.is_as() or self.__played_card.is_three()):
                    if self.__played_card.is_same_suit(self.__initial_card.get_suit_id()):
                        # La guanya amb l'As o el 3 del coll de ronda
                        # Sumem 10 punts
                        heuristic += 10
                    else:
                        # La guanya amb l'As o el 3 del coll de triomf
                        # Restem 10 punts
                        heuristic -= 10
                elif self.__can_win_with_suit and self.__played_card.is_same_suit(self.__initial_card.get_suit_id()):
                    # La guanya amb qualsevol carta del coll ronda
                    heuristic += 8
                elif self.__can_win_with_trump and self.__played_card.is_same_suit(self.__trump.get_suit_id()):
                    # La guanya amb qualsevol carta del coll del triomf
                    # Sumem 6 punts
                    heuristic += 6

                if local_print_console:
                    print("heuristic 5", heuristic)
            elif self.__wins_round and self.__can_win_with_high:
                # Guanya ronda on no s'han jugat cartes valuoses
                if (self.__played_card.is_as() or self.__played_card.is_three()) and self.__played_card.is_same_suit(self.__initial_card.get_suit_id()):
                    # La guanya amb l'As o el 3 del coll de ronda
                    # Sumem 10 punts
                    heuristic += 10
                else:
                    # La guanya sense l'As o el 3 del coll de ronda, quan en té un d'ells
                    # Restem 10 punts
                    heuristic -= 10
                if local_print_console:
                    print("heuristic 6", heuristic)
            elif self.__wins_round and self.__played_card.is_same_suit(self.__trump.get_suit_id()) and self.__has_lower_cards_trump_suit_to_win_round:
                # Guanya ronda on no s'han jugat cartes valuoses amb un triomf, tenint cartes de triomf de menys valor
                # Restem la diferencia de valor
                heuristic -= self.__played_card.get_value() - self.__lower_card_trump_suit_to_win_round.get_value()
                if local_print_console:
                    print("heuristic 6.5", heuristic)

            # No es guanya la ronda, jugar cartes de valor tenint sense valor a la mà o amb cartes que podria haver guanyat la ronda és incorrecte (es resta la puntuació de la carta jugada)
            if not self.__wins_round and not self.__teammate_wins_round and (self.__has_cards_with_less_value or self.__can_win_with_high or self.__can_win_with_suit or self.__can_win_with_trump):
                heuristic -= self.__played_card.get_value()
                if local_print_console:
                    print("heuristic 7", heuristic)

            # No hi ha cartes de valor jugades i juga triomf tenint altres cartes sense valor a la mà, restem 30 punts
            if not self.__valuable_cards_played and self.__has_low_cards_not_trump_suit and self.__played_card.is_same_suit(self.__trump.get_suit_id()):
                heuristic -= 30
                if local_print_console:
                    print("heuristic 8", heuristic)

            # No pot guanyar la ronda i juga triomf, tenint altres cartes sense valor a la mà
            # Restem 5 punts
            if not self.__can_win_with_high and not self.__can_win_with_trump and not self.__can_win_with_high and self.__has_low_cards_not_trump_suit and self.__played_card.is_same_suit(self.__trump.get_suit_id()):
                heuristic -= 5
                if local_print_console:
                    print("heuristic 9", heuristic)

        # S'està utilitzant un rei o cavall quan es pot utilitzar qualsevol altra carta
        # Error greu, restem 10 punts
        if not self.__is_brisca and self.__can_use_other_than_suit_king_knight and (self.__played_card.is_same_suit(self.__self_singed_suit) and (self.__played_card.is_king() or self.__played_card.is_knight())):
            heuristic -= 10

        if local_print_console:
            print("heuristic 10", heuristic)

        if local_print_console:
            print("")

        return heuristic

    def heuristics(self) -> int:
        # TODO caldria generar una heurística per a cada modalitat i nombre de jugadors per tal d'optimitzar les jugades

        #         if self.__is_brisca and self.__num_players == 2:
        #             return self.__heuristics_brisca_2j()
        #         elif self.__is_brisca and self.__num_players == 3:
        #             return self.__heuristics_brisca_2j()
        #         elif self.__is_brisca and self.__num_players == 4:
        #             return self.__heuristics_brisca_2j()

        # TODO Revisar heuristics amb 4 jugadors per equips
        # TODO Si el rival ha cantat, el rival ha de jugar a guanyar la ronda si o si (donar punts), sino restar punts

        return self.__heuristics_brisca_2j()
