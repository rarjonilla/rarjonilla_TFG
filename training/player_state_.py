import copy
from typing import List, Optional, Dict

from game_environment.card import Card

import numpy as np


class Player_state:
    """Classe training player state....."""

    def __init__(self, player_id: int, single_mode: bool, is_brisca: bool, rules: Dict, num_cards: int,
                 num_players: int, deck_size: int, trump: Card, score: List[int],
                 sing_declarations: Optional[List[Optional[int]]], model_type: int) -> None:
        self.__is_brisca = is_brisca
        self.__player_id = player_id
        self.__single_mode = single_mode
        self.__num_cards = num_cards
        self.__rules = rules
        self.__num_players = num_players
        self.__deck_size: int = deck_size
        self.__trump: Card = trump
        self.__score: List[int] = score
        self.__sing_declarations: Optional[List[Optional[int]]] = sing_declarations
        self.__model_type = model_type

        self.__heuristics: int = 0

        self.__round_score = 0
        self.__win = 0

        # self.hands: List[List[Card]] = []
        # self.hand_cards: List[int] = [0] * 40
        self.hand_cards = []
        for i in range(num_players):
            hand_card: List[int] = [0] * 40
            self.hand_cards.append(hand_card)

        # cards_list = [0] * 40
        # self.hand_cards = [copy.deepcopy(cards_list)] * num_players

        self.score: List[int] = [0] * num_players
        self.__round_played_cards: List[Optional[Card]] = [] * num_players
        self.all_played_cards: List[int] = []
        # self.singed_declarations: List[int] = singed_declarations

        self.hands: List[List[Card]] = [] * num_players

        # self.action: int = 0
        # Posicions: 0-39 -> vector posicio de carta
        # Posicions: 40 -> can change (si està activa la regla)
        # Posicions: 40-43 //41-44 -> cants tute, depen del nombre de jugadors i de si la regla està activa
        total_actions = 41
        # total_actions += 1 if self.__rules["can_change"] else 0
        total_actions += 0 if is_brisca else 4

        self.actions: List[int] = [0] * total_actions

    def set_round_played_cards(self, played_cards: List[Optional[Card]]) -> None:
        self.__round_played_cards = played_cards

    def set_round_score(self, round_score: int) -> None:
        self.__round_score = round_score

    def set_winner(self) -> None:
        self.__win = 1

    def set_all_played_cards(self, all_played_cards: List[int]) -> None:
        self.all_played_cards = all_played_cards

    # def set_sing_declarations(self, sing_declarations: List[int]) -> None:
    #     self.__sing_declarations = sing_declarations

    def set_hands(self, hands: List[List[Card]]) -> None:
        self.hands = hands

    # type -> 0=canvi, 1=carta, 2=cant
    def set_action(self, action: int, type: int) -> None:
        if type == 0:
            self.actions[40] = 1
        elif type == 1:
            self.actions[action] = 1
        elif type == 2:
            # action_pos = 40 if self.__rules['can_change'] else 39
            action_pos = 40
            action_pos += action
            self.actions[action_pos] = 1

    #     def show(self) -> None:
    #         print("********************************")
    #         print("deck_size", self.__deck_size)
    #         print("trump", self.__trump)
    #         print("")
    #         print("played_cards: ")
    #         jdx: int = 0
    #         for jdx, round_played_card in enumerate(self.__round_played_cards):
    #             print("card " + str(jdx), round_played_card)
    #
    #         for i in range(len(self.__round_played_cards), self.__num_players - 1):
    #             print("card " + str(i) + " None " + str(i))
    #
    #         print("")
    #         for idx, hand in enumerate(self.hands):
    #             print("hand player " + str(idx))
    #             jdx = 0
    #             for jdx, card_in_hand in enumerate(hand):
    #                 print("card " + str(jdx), card_in_hand)
    #
    #             for i in range(len(hand), self.__num_cards):
    #                 print("card " + str(i) + " None " + str(i))
    #
    #         print("")
    #         print("sing_declarations", self.__sing_declarations)
    #
    #         print("")
    #         print("all_played_cards: ")
    #         for jdx, card in enumerate(self.all_played_cards):
    #             print("card " + str(jdx), card)
    #
    #         print("")
    #
    #         print("score", self.__score)
    #         print("action", self.action)

    def __value_to_one_hot_encoding_intervals(self, value, intervals):
        vector = [0] * len(intervals)
        for i, interval in enumerate(intervals):
            if interval[0] <= value <= interval[1]:
                vector[i] = 1
                break

        return vector

    def __value_to_one_hot_encoding(self, value, total_inputs):
        vector = [0] * total_inputs
        vector[value - 1] = 1
        return vector

    def get_inputs_array(self) -> List[int]:
        # TODO -> Segons el model_type, hi haurà inputs que no s'hauran d'afegir
        inputs_array: List[int] = []

        # Deck size -> cada interval representarà una fase del joc
        # Brisca -> 2 jugadors: 0-34
        # 4 intervals (0)(0-6)(7-18)(19-34) // 0 = ultimes rondes, 0-6 = 3 ultimes rondes amb cartes restants
        # Brisca -> 3 jugadors: 0-31
        # 4 intervals (1)(2-10)(11-19)(20-31) // 1 = ultimes rondes, 2-10 = 3 ultimes rondes amb cartes restants
        # Brisca -> 4 jugadors: 0-28
        # 4 intervals (0)(0-8)(9-16)(17-28) // 0 = ultimes rondes, 0-8 = 2 ultimes rondes amb cartes restants

        # Tute -> 2 jugadors: 0-24
        # 3 intervals (0)(0-8)(9-16)(16-24)
        # Tute -> 3 jugadors: 0-16
        # 3 intervals (1)(2-7)(8-16) // 1 = ultimes rondes, 2-7 = 2 ultimes rondes amb cartes restants
        # Tute -> 4 jugadors: 0-8
        # 2 intervals (0)(0-8) // 0 = ultimes rondes, 0-8 = 2 ultimes rondes amb cartes restants
        if self.__model_type == 2 or self.__model_type == 3:
            if self.__is_brisca:
                if self.__num_players == 2:
                    intervals = [(0, 0), (1, 6), (7, 18), (19, 34)]
                elif self.__num_players == 3:
                    intervals = [(0, 1), (2, 10), (11, 19), (20, 31)]
                else:
                    intervals = [(0, 0), (1, 8), (9, 16), (17, 28)]
            else:
                if self.__num_players == 2:
                    intervals = [(0, 0), (1, 8), (9, 16), (16, 24)]
                elif self.__num_players == 3:
                    intervals = [(0, 1), (2, 7), (8, 16)]
                else:
                    intervals = [(0, 0), (1, 8)]

            inputs_array += self.__value_to_one_hot_encoding_intervals(self.__deck_size, intervals)

        # Carta de triomf
        # 4 inputs one shot per el coll: Ors -> (1, 0, 0, 0)
        # 10 inputs one shot per la carta segons preferencia de menor a major (2, 4, 5, ..., As)
        # As (0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
        inputs_array += self.__value_to_one_hot_encoding(self.__trump.get_suit_id(), 4)
        inputs_array += self.__value_to_one_hot_encoding(self.__trump.get_training_value(), 10)

        # Cartes jugades previament
        # Per a cada possible carta jugada previament (num jugadors - 1):
        # 4 inputs one shot per el coll: Ors -> (1, 0, 0, 0)
        # 10 inputs one shot per la carta segons preferencia de menor a major (2, 4, 5, ..., As)
        # As (0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
        # Si en una "posicio" no hi ha carta jugada, tant inputs coll com carta seran "0"
        for round_played_card in self.__round_played_cards:
            if round_played_card is None:
                raise AssertionError("Round played card is None")
            inputs_array += self.__value_to_one_hot_encoding(round_played_card.get_suit_id(), 4)
            inputs_array += self.__value_to_one_hot_encoding(round_played_card.get_training_value(), 10)

        for i in range(len(self.__round_played_cards), self.__num_players - 1):
            inputs_array += [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Mans dels jugadors (el jugador coneix la seva mà i les cartes que ha pogut veure de la resta de rivals)
        # L'ordre és des de la vista del jugador. Primer les seves cartes, després el següent a la seva esquerra, el segon a la seva esquerra, ...
        # Vector one shot de 40 posicions fixes on cada posicio indica una carta. 0 = no té la carta d'aquella posicio, 1 = té la carta de la posició
        # L'ordre és per etiqueta de la carta: As d'ors, 2 d'ors, 3 d'ors, ..., 12 d'ors, As de bastos, ... 12 d'espases, ... 12 de copes
        for np in range(self.__num_players):
            pos: int = self.__player_id + np
            pos %= self.__num_players

            hand = self.hands[pos]
            # print(hand)

            for card_in_hand in hand:
                self.hand_cards[pos][card_in_hand.get_training_idx()] = 1
            for hand_card in self.hand_cards[pos]:
                inputs_array += [hand_card]

        # Cants per a cada jugador
        # L'ordre és des de la vista del jugador. Primer les seves cartes, després el següent a la seva esquerra, el segon a la seva esquerra, ...
        # Cada jugador és un vector one shot de 4 posicions (1 per coll)
        if not self.__is_brisca:
            # Només Tute
            if self.__sing_declarations is None or len(self.__sing_declarations) == 0:
                for i in range(self.__num_players):
                    inputs_array += [0, 0, 0, 0]
            else:
                for np in range(self.__num_players):
                    pos = self.__player_id + np
                    pos %= self.__num_players

                    sing_declaration = self.__sing_declarations[pos]

                    if sing_declaration is None:
                        inputs_array += [0, 0, 0, 0]
                    else:
                        for j in range(1, 5):
                            if sing_declaration == j:
                                inputs_array += [1]
                            else:
                                inputs_array += [0]

        # print("all_played_cards: ")
        # Cartes jugades en tota la partida
        # Vector one shot de 40 posicions fixes on cada posicio indica una carta. 0 = no s'ha jugat la carta d'aquella posicio, 1 = s'ha jugat la carta de la posició
        for is_viewed in self.all_played_cards:
            inputs_array += [is_viewed]

        # Puntuació per a cada jugador
        # L'ordre és des de la vista del jugador. Primer la seva puntuació, després el següent a la seva esquerra, el segon a la seva esquerra, ...
        # 7 intervals discrets per a cada jugador (0-19)(20-39)(40-59)...(100-119)(+120)
        intervals = [(0, 19), (20, 39), (40, 59), (60, 79), (80, 99), (100, 119), (120, 300)]
        for np in range(self.__num_players):
            pos = self.__player_id + np
            pos %= self.__num_players

            player_score = self.__score[pos]

            inputs_array += self.__value_to_one_hot_encoding_intervals(player_score, intervals)

        # Vector de regles
        # Cada regla és un input que serà 1 si està activada i 0 si no ho està
        inputs_array += [1] if self.__rules['can_change'] else [0]
        inputs_array += [1] if self.__rules['last_tens'] else [0]
        inputs_array += [1] if self.__rules['black_hand'] else [0]

        if self.__num_players == 2:
            inputs_array += [1] if self.__rules['hunt_the_three'] else [0]

        if 1 < self.__model_type < 7:
            # Acció del jugador (només per supervisada)
            # Vector one shot de 41 a 45 posicions fixes on cada posicio indica una carta. 0 = no s'ha jugat la carta, 1 = s'ha jugat la carta
            # total_actions = 41 if self.__is_brisca() else 45
            for action in self.actions:
                inputs_array += [action]

        return inputs_array

    # linia CSV que es guardarà (s'han ajuntat els inputs binaris per blocs i es calcula el seu valor decimal per ocupar menys espai)
    def csv_line(self) -> str:
        # Deck size
        csv_line = str(self.__deck_size) + ","

        # Trump
        # 2 inputs -> coll i label
        csv_line += str(self.__trump.get_suit_id()) + "," + str(self.__trump.get_training_value()) + ","

        # csv_line += "\n"
        # csv_line += "--"

        # Cartes jugades previament
        # Per a cada possible carta jugada previament (num jugadors - 1):
        #   2 inputs -> coll i label
        # Si en una "posicio" no hi ha carta jugada, tant input coll com label seran "0"
        for round_played_card in self.__round_played_cards:
            if round_played_card is None:
                raise AssertionError("Round played card is None")
            csv_line += str(round_played_card.get_suit_id()) + "," + str(round_played_card.get_training_value()) + ","

        for i in range(len(self.__round_played_cards), self.__num_players - 1):
            csv_line += "0,0,"

        # csv_line += "--"
        # csv_line += "\n"

        # Mans dels jugadors (el jugador coneix la seva mà i les cartes que ha pogut veure de la resta de rivals)
        # L'ordre és des de la vista del jugador. Primer les seves cartes, després el següent a la seva esquerra, el segon a la seva esquerra, ...
        # Per a cada carta:
        #   Vector one shot de 40 posicions fixes on cada posicio indica una carta. 0 = no té la carta d'aquella posicio, 1 = té la carta de la posició
        # Es reconverteix la mà de cada jugador (els 40 inputs a nombre decimal per ocupar menys espai)

        for np in range(self.__num_players):
            pos: int = self.__player_id + np
            pos %= self.__num_players

            hand = self.hands[pos]
            binary_hand = ""
            for card_in_hand in hand:
                self.hand_cards[pos][card_in_hand.get_training_idx()] = 1
            for hand_card in self.hand_cards[pos]:
                binary_hand += str(hand_card)

            csv_line += str(int(binary_hand, 2)) + ","

            #             if pos == self.__player_id:
            #                 for jsx, card_in_hand in enumerate(hand):
            #                     csv_line += str(card_in_hand.get_suit_id()) + ", " + str(card_in_hand.get_label()) + ", "
            #
            #                 for jsx in range(len(hand), self.__num_cards + 1):
            #                     csv_line += "0, 0, "
            #             else:
            #                 binary_hand = ""
            #                 for jsx in range(len(hand), self.__num_cards + 1):
            #                     binary_hand += "00"
            #
            #                 for jsx, card_in_hand in enumerate(hand):
            #                     binary_hand += "00"
            #                     csv_line += str(card_in_hand.get_suit_id()) + ", " + str(card_in_hand.get_label()) + ", "

        # csv_line += "\n"
        # csv_line += "--"

        # Cants per a cada jugador
        # L'ordre és des de la vista del jugador. Primer les seves cartes, després el següent a la seva esquerra, el segon a la seva esquerra, ...
        # 4 inputs amb el id del coll, cada input representa un jugador
        if not self.__is_brisca:
            # Només Tute
            if self.__sing_declarations is None or len(self.__sing_declarations) == 0:
                for i in range(self.__num_players):
                    csv_line += "0,"
            else:
                for np in range(self.__num_players):
                    pos = self.__player_id + np
                    pos %= self.__num_players

                    sing_declaration = self.__sing_declarations[pos]

                    if sing_declaration is None:
                        csv_line += "0,"
                    else:
                        csv_line += str(sing_declaration) + ","

            # csv_line += "\n"
            # csv_line += "--"

        # Cartes jugades en tota la partida
        # vector de 40 posicions tractat com nombre decimal (ocupara menys espai)
        binary_played_cards = ""
        for is_viewed in self.all_played_cards:
            binary_played_cards += str(is_viewed)

        csv_line += str(int(binary_played_cards, 2)) + ","

        # csv_line += "\n"
        # csv_line += "--"

        # Puntuació per a cada jugador
        # L'ordre és des de la vista del jugador. Primer la seva puntuació, després el següent a la seva esquerra, el segon a la seva esquerra, ...
        for np in range(self.__num_players):
            pos = self.__player_id + np
            pos %= self.__num_players

            player_score = self.__score[pos]

            csv_line += str(player_score) + ","

        # csv_line += "\n"
        # csv_line += "--"

        # Vector de regles
        # Cada regla és un input que serà 1 si està activada i 0 si no ho està
        # Es tracta el conjunt de regles com un decimal
        # TODO -> A tenir en compte: quan el tradueixi a binari haig de tenir en compte el total de regles que poden haver actives alhora segons la modalitat i nombre de jugadors
        binary_rules = ""
        binary_rules += "1" if self.__rules['can_change'] else "0"
        binary_rules += "1" if self.__rules['last_tens'] else "0"
        binary_rules += "1" if self.__rules['black_hand'] else "0"

        if self.__num_players == 2:
            binary_rules += "1" if self.__rules['hunt_the_three'] else "0"

        # Es tractarà com un model diferent
        # if not self.__is_brisca:
        #     binary_rules += "1" if self.__rules['only_assist'] else "0"

        csv_line += str(int(binary_rules, 2)) + ","
        # csv_line += "--"

        # csv_line += "\n"

        # Acció del jugador (només per supervisada)
        # Vector one shot de 40 a 45 posicions fixes on cada posicio indica una acció. 0 = no s'ha triat l'acció, 1 = s'ha triat l'acció
        # ES converteix a decimal per ocupar menys espai
        # csv_line += "\n---"
        binary_actions = ""
        for action in self.actions:
            binary_actions += str(action)
            # csv_line += str(action) + ", "

        # csv_line += "---\n"
        csv_line += str(int(binary_actions, 2)) + ","
        # csv_line += "--"
        # csv_line += "---\n"

        # TODO -> Falta afegir l'opcio d'intercanvi (1 input)
        # TODO -> Falta afegir la decisió de cant per al tute (4 inputs)

        # Puntuació de ronda, heuristic i win partida
        csv_line += str(self.__round_score) + "," + str(self.__heuristics) + "," + str(self.__win)

        # csv_line += "\n"
        # csv_line += "\n"

        return csv_line

    def csv_line_(self) -> str:
        csv_line = ""

        #         csv_line = self.__trump.get_one_hot()

        #         for round_played_card in self.__round_played_cards:
        #             if round_played_card is None:
        #                 raise AssertionError("Round played card is None")
        #             # csv_line += str(round_played_card.get_suit_id()) + ", " + str(round_played_card.get_training_value()) + ", "
        #             csv_line += round_played_card.get_one_hot()
        #
        #         for i in range(len(self.__round_played_cards), self.__num_players - 1):
        #             csv_line += "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "

        #         for np in range(self.__num_players):
        #             pos: int = self.__player_id + np
        #             pos %= self.__num_players
        #
        #             hand = self.hands[pos]
        #             # print(hand)
        #
        #             for card_in_hand in hand:
        #                 self.hand_cards[pos][card_in_hand.get_training_idx()] = 1
        #             for hand_card in self.hand_cards[pos]:
        #                 csv_line += str(hand_card) + ", "

        #         if not self.__is_brisca:
        #             # Només Tute
        #             if self.__sing_declarations is None:
        #                 for i in range(self.__num_players):
        #                     csv_line += "0, 0, 0, 0, "
        #             else:
        #                 for np in range(self.__num_players):
        #                     pos: int = self.__player_id + np
        #                     pos %= self.__num_players
        #
        #                     sing_declaration = self.__sing_declarations[pos]
        #
        #                     if sing_declaration is None:
        #                         csv_line += "0, 0, 0, 0, "
        #                     else:
        #                         for j in range(1, 5):
        #                             if sing_declaration == j:
        #                                 csv_line += "1, "
        #                             else:
        #                                 csv_line += "0, "

        #         for is_viewed in self.all_played_cards:
        #             csv_line += str(is_viewed) + ", "

        #         for np in range(self.__num_players):
        #             pos: int = self.__player_id + np
        #             pos %= self.__num_players
        #
        #             player_score = self.__score[pos]
        #
        #             csv_line += str(player_score) + ", "

        #         csv_line += "1, " if self.__rules['can_change'] else "0, "
        #         csv_line += "1, " if self.__rules['last_tens'] else "0, "
        #         csv_line += "1, " if self.__rules['black_hand'] else "0, "
        #
        #         if self.__num_players == 2:
        #             csv_line += "1, " if self.__rules['hunt_the_three'] else "0, "
        #
        #         if not self.__is_brisca:
        #             csv_line += "1, " if self.__rules['only_assist'] else "0, "
        #
        #         csv_line += str(self.action) + ""

        return csv_line

    # La idea és sumar aquest valor heuristic al resultat final de la partida (guanyar la partida seran 30 punts extra a l'heuristica de cada ronda)
    # D'aquesta manera, tot i no guanyar la partida s'haurà qualificat la jugada
    # Explicar que per "round points" no funciona bé (juga cartes molt altes sempre per la aleatorietat de les dades inicials (com que la màquina no juga bé, es donen molts casos on aquesta estrategia fa sumar molts punts, però en una partida real això no funciona))
    # Explicar que per "win or lose" no funciona bé (les jugades que porten a guanyar poden ser molt dolentes per la aleatorietat de les dades inicials (com que la màquina no juga bé, es donen molts casos on jugades dolentes donen lloc a guanyar partides, però en una partida real això no funciona))
    # TODO 1 -> Potser s'hauria de fer una prova només amb heuristics negatius (penalitzar més les males accions)
    # TODO 2
    # si l'heuristica funciona, es podrien generar dades aleatories amb aquest model per tal d'entrenar-ne un de nou únicament per "round points" i "win or lose"
    # Aquest primer model amb l'heuristica representaria un conjunt de dades de jugadors semiprofessionals
    def heuristics(self, played_card: Card, playable_hand: List[Card]):
        heuristic: int = 0

        # Afegeixo la carta jugada a playable_hand per fer les comprovacions
        playable_hand.append(played_card)

        is_first_turn: bool = True if len(self.__round_played_cards) == 0 else False

        # Calcul de les diferents possibilitats de les cartes jugades que ajudaran al càlcul d'heuristics final
        initial_card: Optional[Card] = None if is_first_turn else self.__round_played_cards[0]

        highest_initial_suit_label: int = 0 if is_first_turn else initial_card.get_label()
        highest_initial_suit_value: int = 0 if is_first_turn else initial_card.get_value()

        is_initial_suit_trump: bool = True if not is_first_turn and self.__trump.is_same_suit(
            initial_card.get_suit_id()) else False
        trump_used: bool = is_initial_suit_trump

        highest_trump_label: int = 0 if not is_initial_suit_trump else highest_initial_suit_label
        highest_trump_value: int = 0 if not is_initial_suit_trump else highest_initial_suit_value
        highest_trump: int = None if not is_initial_suit_trump else initial_card

        valuable_cards_played: bool = False
        wins_round: bool = False
        teammate_wins_round: bool = False

        teammate_id = (self.__player_id + 2) % self.__num_players
        teammate_played_position: int = -1

        if not self.__single_mode and len(self.__round_played_cards) == 2:
            teammate_played_position = 0
        elif not self.__single_mode and len(self.__round_played_cards) == 3:
            teammate_played_position = 1

        # print("teammate_played_position", teammate_played_position)

        # Cartes ja jugades
        for played_c in self.__round_played_cards:
            if self.__trump.is_same_suit(played_c.get_suit_id()):
                trump_used = True
                if played_c.has_higher_value(highest_trump_value) or (
                        played_c.has_same_value(highest_trump_value) and played_c.is_higher_label(highest_trump_label)):
                    highest_trump_label = played_c.get_label()
                    highest_trump_value = played_c.get_value()
                    highest_trump = played_c

            if played_c.is_same_suit(initial_card.get_suit_id()) and played_c.has_higher_value(
                    highest_initial_suit_value) or (
                    played_c.has_same_value(highest_initial_suit_value) and played_c.is_higher_label(
                    highest_initial_suit_label)):
                highest_initial_suit_label = played_c.get_label()
                highest_initial_suit_value = played_c.get_value()

            if played_c.is_as() or played_c.is_three():
                valuable_cards_played = True

        if not is_first_turn and not trump_used and played_card.is_same_suit(initial_card.get_suit_id()) and (
                played_card.has_higher_value(highest_initial_suit_value) or (
                played_card.has_same_value(highest_initial_suit_value) and played_card.is_higher_label(
                highest_initial_suit_label))):
            wins_round = True
        elif not is_first_turn and not trump_used and played_card.is_same_suit(self.__trump.get_suit_id()) and (
                played_card.has_higher_value(highest_trump_value) or (
                played_card.has_same_value(highest_trump_value) and played_card.is_higher_label(highest_trump_label))):
            wins_round = True
        elif not is_first_turn and trump_used and played_card.is_same_suit(self.__trump.get_suit_id()) and (
                played_card.has_higher_value(highest_trump_value) or (
                played_card.has_same_value(highest_trump_value) and played_card.is_higher_label(highest_trump_label))):
            wins_round = True

        teammate_played_card = None
        if teammate_played_position != -1:
            teammate_played_card = self.__round_played_cards[teammate_played_position]

            if not trump_used and teammate_played_card.is_same_suit(
                    initial_card.get_suit_id()) and teammate_played_card.has_same_value(
                    highest_initial_suit_value) and teammate_played_card.get_label() == highest_initial_suit_label:
                teammate_wins_round = True
            elif not trump_used and teammate_played_card.is_same_suit(
                    self.__trump.get_suit_id()) and teammate_played_card.has_same_value(
                    highest_trump_value) and teammate_played_card.get_label() == highest_trump_label:
                teammate_wins_round = True
            elif trump_used and teammate_played_card.is_same_suit(
                    self.__trump.get_suit_id()) and teammate_played_card.has_same_value(
                    highest_trump_value) and teammate_played_card.get_label() == highest_trump_label:
                teammate_wins_round = True

        # print("teammate_wins_round", teammate_wins_round)

        # As i 3 jugats dels colls
        high_cards_played: List[int] = []
        for i in range(0, 4):
            total_high_played: int = 0
            position_as: int = 0 + 10 * i
            position_three: int = 2 + 10 * i

            total_high_played += 1 if self.all_played_cards[position_as] == 1 else 0
            total_high_played += 1 if self.all_played_cards[position_three] == 1 else 0

            high_cards_played.append(total_high_played)

        max_played_as_and_three_suit = max(high_cards_played)

        # Tute
        is_self_sing_suit: bool = False
        is_team_sing_suit: bool = False
        has_other_sing_suit: bool = False
        is_other_sing_suit: bool = False
        can_use_other_than_suit_king_knight: bool = False
        self_singed_suit: int = None
        team_singed_suit: int = None

        if self.__sing_declarations is not None:
            self_singed_suit = self.__sing_declarations[self.__player_id]
            if not self.__single_mode and teammate_played_position != -1:
                team_singed_suit = self.__sing_declarations[teammate_played_position]

            if self_singed_suit is not None and played_card.is_same_suit(self_singed_suit):
                is_self_sing_suit = True

            if team_singed_suit is not None and played_card.is_same_suit(team_singed_suit):
                is_team_sing_suit = True

            for i in range(0, 4):
                if i != self.__player_id:
                    singed_suit = self.__sing_declarations[i]
                    if singed_suit is not None and played_card.is_same_suit(singed_suit):
                        is_other_sing_suit = True

        # Cartes a la mà del jugador
        can_win_with_suit: bool = False
        can_win_with_high: bool = False
        can_win_with_trump: bool = False
        has_trump_as: bool = False
        has_cards_with_less_value: bool = False
        has_cards_of_most_played_as_and_three_suit: bool = False
        has_low_cards_not_trump_suit: bool = False
        has_more_valuable_card: bool = False
        more_valuable_card: Card = None

        has_lower_cards_trump: bool = False
        lower_card_trump: Card = None
        has_lower_cards_trump_suit_to_win_round: bool = False
        lower_card_trump_suit_to_win_round: Card = None

        for card_in_hand in playable_hand:
            if played_card.is_same_suit(self.__trump.get_suit_id()) and card_in_hand.is_same_suit(
                    self.__trump.get_suit_id()) and played_card.has_higher_value(card_in_hand.get_value()):

                if lower_card_trump is None or lower_card_trump.has_higher_value(card_in_hand.get_value()):
                    has_lower_cards_trump = True
                    lower_card_trump = card_in_hand

                if not trump_used:
                    has_lower_cards_trump_suit_to_win_round = True
                    if lower_card_trump_suit_to_win_round is None or lower_card_trump_suit_to_win_round.has_higher_value(
                            card_in_hand.get_value()):
                        lower_card_trump_suit_to_win_round = card_in_hand
                elif trump_used and (card_in_hand.has_higher_value(highest_trump_value) or (
                        card_in_hand.has_same_value(highest_trump_value) and card_in_hand.is_higher_label(
                        highest_trump_label))):
                    has_lower_cards_trump_suit_to_win_round = True
                    if lower_card_trump_suit_to_win_round is None or lower_card_trump_suit_to_win_round.has_higher_value(
                            card_in_hand.get_value()):
                        lower_card_trump_suit_to_win_round = card_in_hand

            if not is_first_turn:
                if not trump_used and card_in_hand.is_same_suit(initial_card.get_suit_id()) and (
                        card_in_hand.is_as() or card_in_hand.is_three()) and (
                        card_in_hand.has_higher_value(highest_initial_suit_value) or (
                        card_in_hand.has_same_value(highest_initial_suit_value) and card_in_hand.is_higher_label(
                        highest_initial_suit_label))):
                    can_win_with_high = True

                if not trump_used and card_in_hand.is_same_suit(initial_card.get_suit_id()) and (
                        card_in_hand.has_higher_value(highest_initial_suit_value) or (
                        card_in_hand.has_same_value(highest_initial_suit_value) and card_in_hand.is_higher_label(
                        highest_initial_suit_label))):
                    can_win_with_suit = True

            if card_in_hand.is_same_suit(self.__trump.get_suit_id()) and (
                    card_in_hand.has_higher_value(highest_trump_value) or (
                    card_in_hand.has_same_value(highest_trump_value) and card_in_hand.is_higher_label(
                    highest_trump_label))):
                can_win_with_trump = True

            if card_in_hand.is_same_suit(self.__trump.get_suit_id()) and card_in_hand.is_as():
                has_trump_as = True

            if played_card.has_higher_value(card_in_hand.get_value()):
                has_cards_with_less_value = True

            # print(not card_in_hand.is_same_suit(self.__trump.get_suit_id()), high_cards_played[card_in_hand.get_suit_id() - 1], max_played_as_and_three_suit)
            if not card_in_hand.is_same_suit(self.__trump.get_suit_id()) and high_cards_played[
                card_in_hand.get_suit_id() - 1] == max_played_as_and_three_suit:
                has_cards_of_most_played_as_and_three_suit = True

            if not card_in_hand.is_same_suit(self.__trump.get_suit_id()) and card_in_hand.get_value() < 4:
                has_low_cards_not_trump_suit = True

            if not card_in_hand.is_same_suit(self.__trump.get_suit_id()) and card_in_hand.has_higher_value(
                    played_card.get_value()):
                has_more_valuable_card = True
                if more_valuable_card is None or card_in_hand.has_higher_value(more_valuable_card.get_value()):
                    more_valuable_card = card_in_hand

            if self.__sing_declarations is not None:
                for i in range(0, 4):
                    if i != self.__player_id:
                        singed_suit = self.__sing_declarations[i]
                        # print("---------")
                        # print(singed_suit)
                        # print(card_in_hand)

                        if singed_suit is not None and card_in_hand.is_same_suit(singed_suit):
                            has_other_sing_suit = True

            if self_singed_suit is not None and not played_card.is_king() and not played_card.is_knight() and played_card.is_same_suit(
                    self_singed_suit):
                can_use_other_than_suit_king_knight = True
        #         print("")
        #         print("is_first_turn", is_first_turn)
        #         print("highest_initial_suit_label", highest_initial_suit_label)
        #         print("highest_initial_suit_value", highest_initial_suit_value)
        #         print("is_initial_suit_trump", is_initial_suit_trump)
        #         print("trump_used", trump_used)
        #         print("highest_trump_label", highest_trump_label)
        #         print("highest_trump_value", highest_trump_value)
        #         print("valuable_cards_played", valuable_cards_played)
        #         print("wins_round", wins_round)
        # print("high_cards_played", high_cards_played)
        # print("max_played_as_and_three_suit", max_played_as_and_three_suit)
        # print("has_cards_of_most_played_as_and_three_suit", has_cards_of_most_played_as_and_three_suit)
        #         print("can_win_with_suit", can_win_with_suit)
        #         print("can_win_with_high", can_win_with_high)
        #         print("can_win_with_trump", can_win_with_trump)
        #         print("has_trump_as", has_trump_as)
        #         print("has_cards_with_less_value", has_cards_with_less_value)
        #         print("playable_hand", len(playable_hand))
        # print("has_more_valuable_card", has_more_valuable_card)
        #         print("")

        print("*********************")

        if is_first_turn:
            # print("first turn")
            # print(self.__round_played_cards)
            # Inici de torn

            # ****
            if (played_card.is_as() or played_card.is_three()) and has_low_cards_not_trump_suit:
                # Iniciar el torn amb As o 3 no és correcte (de vegades és bo depenent si sospites que el rival no te cartes de triomf, però, en general, és una jugada horrible)
                # Restem el valor de la carta per 2
                # heuristic = -3 if played_card.is_as() or played_card.is_three() else 3
                heuristic -= played_card.get_value() * 2
                print("heuristic 1", heuristic)

            # ****
            # Juga carta de triomf amb valor, tenint cartes de triomf de menys valor
            # Restem la diferencia del valor de la carta
            if played_card.is_same_suit(self.__trump.get_suit_id()) and has_lower_cards_trump:
                # Si s'inicia una ronda amb triomf, que sigui la mes baixa
                # Restem la diferencia del valor de les carta per 2
                # heuristic = -3 if played_card.is_as() or played_card.is_three() else 3
                heuristic -= played_card.get_value() - lower_card_trump.get_value()
                print("heuristic 1.5", heuristic)

            # ****
            # Iniciar el torn amb carta del coll del triomf, tenint altres cartes de baix valor no és correcte
            # Restarem 10 punts2
            if played_card.is_same_suit(self.__trump.get_suit_id()) and has_low_cards_not_trump_suit:
                # heuristic -= 3
                heuristic -= 10
                print("heuristic 2", heuristic)

            # ****
            # S'ha de prioritzar utilitzar cartes de colls on ja s'hagin jugat el 3 i l'As (que no siguin triomf), si s'en disposa
            # Es sumaran / restaran 5 punts
            t: int = high_cards_played[played_card.get_suit_id() - 1]
            if max_played_as_and_three_suit > 0:
                if has_cards_of_most_played_as_and_three_suit and not t == max_played_as_and_three_suit:
                    # heuristic -= 2
                    heuristic -= 5
                elif t == max_played_as_and_three_suit and not played_card.is_same_suit(self.__trump.get_suit_id()):
                    # heuristic += 2
                    # Sumem només si no es tracta del coll del triomf
                    heuristic += 5

                print("heuristic 3", heuristic)

            # S'esta utilitzant una carta del coll del qual el propi jugador o el seu company han cantat
            # Error molt greu, restem 10 punts
            if not self.__is_brisca and (is_self_sing_suit or is_team_sing_suit):
                heuristic -= 10

            # print(is_other_sing_suit, has_other_sing_suit, is_self_sing_suit, is_team_sing_suit)
            if not self.__is_brisca and is_other_sing_suit:
                # S'esta utilitzant una carta del coll del qual un rival ha cantat
                # Bona jugada, sumem 10 punts
                heuristic += 10
            elif not self.__is_brisca and not is_other_sing_suit and has_other_sing_suit and not is_self_sing_suit and not is_team_sing_suit:
                # S'esta utilitzant una carta que no és del coll cantat pel rival
                # Error molt greu, restem 10 punts (només si es disposa de carta del coll cantat)
                heuristic -= 10

            print("heuristic 4", heuristic)


        # elif len(self.__round_played_cards) == self.__num_players - 1:
        # Final de torn
        else:
            # ****
            # Si s'ha jugat el 3 del coll del triomf i el jugador té l'As, jugar-lo és correcte, no jugar-lo incorrecte
            # Sumem / restem 5 punts
            if highest_trump is not None and highest_trump.is_three() and played_card.is_same_suit(
                    self.__trump.get_suit_id()) and played_card.is_as():
                heuristic += 5
            elif highest_trump is not None and highest_trump.is_three() and has_trump_as:
                heuristic -= 5

            print("heuristic 1", heuristic)

            if len(self.__round_played_cards) == self.__num_players - 1 and not self.__single_mode:
                if teammate_wins_round and has_more_valuable_card:
                    # Si el company guanya la ronda, s'ha de jugar la carta amb més valor de la mà que no sigui triomf
                    # En aquest cas està jugant una carta amb menys valor de les disponibles
                    # Restem la diferència del valor de les cartes (no és el mateix jugar un 3 tenint un As on es perd 1 punt que jugar un 2 tenint un As on es perden 11 punts)
                    # heuristic -= 10
                    # heuristic -= more_valuable_card.get_value()
                    heuristic -= more_valuable_card.get_value() - played_card.get_value()
                elif teammate_wins_round and not has_more_valuable_card:
                    # Si juga la carta amb més valor, afegirem aquests punts extra per tal de bonificar encara més aquesta jugada
                    heuristic += played_card.get_value()

                print("heuristic 2", heuristic)

            # ****
            if not wins_round and valuable_cards_played:
                # No guanya ronda on s'han jugat cartes valuoses quan podria haver-ho fet
                if not teammate_wins_round:
                    if can_win_with_high:
                        # Podia guanyar la ronda amb un As o 3, restem 10
                        heuristic -= 20
                    elif can_win_with_suit:
                        # Podia guanyar la ronda amb qualsevol carta del col de ronda, restem 4
                        heuristic -= 16
                    elif can_win_with_trump:
                        # Podia guanyar la ronda amb qualsevol carta de triomf, restem 10
                        heuristic -= 16

                    print("heuristic 3 if", heuristic)
                else:
                    # Cas on el company està guanyant la ronda
                    # print("+++++++++++++++++")
                    if can_win_with_high:
                        # Podiem guanyar la ronda amb un As o 3 del coll de ronda, restem 10 punts
                        heuristic -= 20

                    print("heuristic 3 else", heuristic)
            elif not wins_round and can_win_with_high:
                if can_win_with_high:
                    # No guanya ronda on no s'han jugat cartes valuoses, però té As o 3 del coll d'inici
                    # Restem 10 punts
                    # heuristic -= 8
                    heuristic -= 20
                print("heuristic 4", heuristic)
            elif wins_round and valuable_cards_played:
                # Guanya ronda on s'han jugat cartes valuoses
                if can_win_with_high:
                    # La guanya amb l'As o el 3 del coll de ronda
                    # Sumem 10 punts
                    heuristic += 20
                elif can_win_with_suit and played_card.is_same_suit(initial_card.get_suit_id()):
                    # La guanya amb qualsevol carta del coll ronda
                    heuristic += 16
                elif can_win_with_trump and played_card.is_same_suit(self.__trump.get_suit_id()):
                    # La guanya amb qualsevol carta del coll del triomf
                    # Sumem 10 punts
                    heuristic += 16

                print("heuristic 5", heuristic)
            elif wins_round and can_win_with_high:
                # Guanya ronda on no s'han jugat cartes valuoses
                if can_win_with_high and played_card.is_as() or played_card.is_three():
                    # La guanya amb l'As o el 3 del coll de ronda
                    # Sumem 10 punts
                    heuristic += 20
                else:
                    # La guanya sense l'As o el 3 del coll de ronda, quan en té un d'ells
                    # Restem 10 punts
                    heuristic -= 10

                print("heuristic 6", heuristic)

            # ****
            # No es guanya la ronda, jugar cartes de valor tenint sense valor a la mà o amb cartes que podria haver guanyat la ronda és incorrecte (es resta la puntuació de la carta jugada)
            # if not wins_round and has_cards_with_less_value and played_card.has_higher_value(4):
            if not wins_round and not teammate_wins_round and (
                    has_cards_with_less_value or can_win_with_high or can_win_with_suit or can_win_with_trump):
                # heuristic -= 10
                heuristic -= played_card.get_value()
                print("heuristic 7", heuristic)

            # ****
            # No hi ha cartes de valor jugades i juga triomf tenint altres cartes sense valor a la mà, restem 30 punts
            if not valuable_cards_played and has_low_cards_not_trump_suit and played_card.is_same_suit(
                    self.__trump.get_suit_id()):
                # heuristic -= 3
                heuristic -= 30
                print("heuristic 8", heuristic)

            # No pot guanyar la ronda i juga triomf, tenint altres cartes sense valor a la mà
            # Restem 5 punts
            if not can_win_with_high and not can_win_with_trump and not can_win_with_high and has_low_cards_not_trump_suit and played_card.is_same_suit(
                    self.__trump.get_suit_id()):
                # heuristic -= 3
                heuristic -= 5
                print("heuristic 9", heuristic)

        # S'està utilitzant un rei o cavall quan es pot utilitzar qualsevol altra carta
        # Error greu, restem 5 punts
        if not self.__is_brisca and can_use_other_than_suit_king_knight and (
                played_card.is_same_suit(self_singed_suit) and (played_card.is_king() or played_card.is_knight())):
            # heuristic -= 10
            heuristic -= 5

        print("heuristic 10", heuristic)

        self.__heuristics = heuristic

        heuristic
