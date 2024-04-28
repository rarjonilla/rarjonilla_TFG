from typing import List, Optional, Dict

from game_environment.card import Card

from training.supervised.player_state_heuristic import Player_state_heuristic


class Player_state:
    """Classe training player state....."""

    def __init__(self, player_id: int, single_mode: bool, is_brisca: bool, rules: Dict, num_cards: int, num_players: int, deck_size: int, trump: Card, score: List[int], sing_declarations: Optional[List[Optional[int]]], model_type: int) -> None:
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
            # Es dona prioritat a l'intercanvi
            # self.__heuristics = 30
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

    def __min_max_normalize(self, value, min_value: int, max_value: int):
        return (value - min_value) / (max_value - min_value)

    def get_inputs_array(self) -> List[int]:
        # TODO -> Segons el model_type, hi haurà inputs que no s'hauran d'afegir
        # print(self.__model_type)
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
        # TODO -> Eliminar 2 a 5 (només utilitzo la 6), fer que sigui la 2 i prou
        # TODO -> Genètic passarà a ser la 3
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
        if self.__model_type == 8:
            # 2 inputs normalitzats
            # inputs_array.append(self.__min_max_normalize(self.__trump.get_suit_id(), 1, 4))
            inputs_array += self.__value_to_one_hot_encoding(self.__trump.get_suit_id(), 4)
            inputs_array.append(self.__min_max_normalize(self.__trump.get_training_value(), 1, 10))
        else:
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

            if self.__model_type == 8:
                # 2 inputs normalitzats
                # inputs_array.append(self.__min_max_normalize(round_played_card.get_suit_id(), 0, 4))
                inputs_array += self.__value_to_one_hot_encoding(round_played_card.get_suit_id(), 4)
                inputs_array.append(self.__min_max_normalize(round_played_card.get_training_value(), 0, 10))
            else:
                inputs_array += self.__value_to_one_hot_encoding(round_played_card.get_suit_id(), 4)
                inputs_array += self.__value_to_one_hot_encoding(round_played_card.get_training_value(), 10)

        for i in range(len(self.__round_played_cards), self.__num_players - 1):
            if self.__model_type == 8:
                # 2 inputs normalitzats
                # inputs_array.append(0)
                inputs_array += [0, 0, 0, 0]
                inputs_array.append(0)
            else:
                inputs_array += [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Mans dels jugadors (el jugador coneix la seva mà i les cartes que ha pogut veure de la resta de rivals)
        # L'ordre és des de la vista del jugador. Primer les seves cartes, després el següent a la seva esquerra, el segon a la seva esquerra, ...
        # Vector one shot de 40 posicions fixes on cada posicio indica una carta. 0 = no té la carta d'aquella posicio, 1 = té la carta de la posició
        # L'ordre és per etiqueta de la carta: As d'ors, 2 d'ors, 3 d'ors, ..., 12 d'ors, As de bastos, ... 12 d'espases, ... 12 de copes
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
        #                 inputs_array += [hand_card]

        hand = self.hands[self.__player_id]
        # print(hand)

        binary_hand = ""
        for card_in_hand in hand:
            self.hand_cards[self.__player_id][card_in_hand.get_training_idx()] = 1
        for hand_card in self.hand_cards[self.__player_id]:
            binary_hand += str(hand_card)
            if self.__model_type < 8:
                inputs_array += [hand_card]

        if self.__model_type == 8:
            # 1 input normalitzat, hem de passar les 40 posicions del vector a decimal
            decimal_value = int(binary_hand, 2)
            inputs_array.append(self.__min_max_normalize(decimal_value, 1, 2**40 - 1))

        # Cants per a cada jugador
        # L'ordre és des de la vista del jugador. Primer les seves cartes, després el següent a la seva esquerra, el segon a la seva esquerra, ...
        # Cada jugador és un vector one shot de 4 posicions (1 per coll)
        if not self.__is_brisca:
            # Només Tute
            if self.__sing_declarations is None or len(self.__sing_declarations) == 0:
                for i in range(self.__num_players):
                    if self.__model_type == 8:
                        # 1 input normalitzat
                        inputs_array.append(0)
                    else:
                        inputs_array += [0, 0, 0, 0]
            else:
                for np in range(self.__num_players):
                    pos = self.__player_id + np
                    pos %= self.__num_players

                    sing_declaration = self.__sing_declarations[pos]

                    if sing_declaration is None:
                        if self.__model_type == 8:
                            # 1 input normalitzat
                            inputs_array.append(0)
                        else:
                            inputs_array += [0, 0, 0, 0]
                    else:
                        if self.__model_type == 8:
                            # 1 input normalitzat
                            inputs_array.append(self.__min_max_normalize(sing_declaration, 0, 4))
                        else:
                            for j in range(1, 5):
                                if sing_declaration == j:
                                    inputs_array += [1]
                                else:
                                    inputs_array += [0]

        # print("all_played_cards: ")
        # Cartes jugades en tota la partida

        binary_played_cards = ""
        for is_viewed in self.all_played_cards:
            binary_played_cards += str(is_viewed)
            if self.__model_type < 8:
                # Vector de 40 posicions fixes on cada posicio indica una carta. 0 = no s'ha jugat la carta d'aquella posicio, 1 = s'ha jugat la carta de la posició
                inputs_array += [is_viewed]

        if self.__model_type == 8:
            decimal_value = int(binary_played_cards, 2)
            inputs_array.append(self.__min_max_normalize(decimal_value, 0, 2**40 - 1))

        # Puntuació per a cada jugador
        # L'ordre és des de la vista del jugador. Primer la seva puntuació, després el següent a la seva esquerra, el segon a la seva esquerra, ...
        # Puntuacio dels jugadors (entre 0 y 130 brisca)
        # Puntuacio dels jugadors (entre 0 y 230 tute)
        max_score = 130 if self.__is_brisca else 230
        # 7 intervals discrets per a cada jugador (0-19)(20-39)(40-59)...(100-119)(+120)
        intervals = [(0, 19), (20, 39), (40, 59), (60, 79), (80, 99), (100, 119), (120, 300)]
        for np in range(self.__num_players):
            pos = self.__player_id + np
            pos %= self.__num_players
            player_score = self.__score[pos]

            if self.__model_type == 8:
                # 1 input normalitzat
                inputs_array.append(self.__min_max_normalize(player_score, 0, max_score))
            else:
                inputs_array += self.__value_to_one_hot_encoding_intervals(player_score, intervals)

        binary_rules = ""
        # Vector de regles
        if self.__model_type == 8:
            binary_rules += "1" if self.__rules['can_change'] else "0"
            binary_rules += "1" if self.__rules['last_tens'] else "0"
            binary_rules += "1" if self.__rules['black_hand'] else "0"

            if self.__num_players == 2:
                binary_rules += "1" if self.__rules['hunt_the_three'] else "0"
            # 1 input normalitzat
            total_rules = 2**4 - 1 if self.__num_players == 2 else 2**3 - 1
            inputs_array.append(self.__min_max_normalize(int(binary_rules, 2), 0, total_rules))
        else:
            # Cada regla és un input que serà 1 si està activada i 0 si no ho està
            inputs_array += [1] if self.__rules['can_change'] else [0]
            inputs_array += [1] if self.__rules['last_tens'] else [0]
            inputs_array += [1] if self.__rules['black_hand'] else [0]

            if self.__num_players == 2:
                inputs_array += [1] if self.__rules['hunt_the_three'] else [0]

        if 1 < self.__model_type < 7 or self.__model_type == 8:
            # Acció del jugador (només per supervisada)
            # binary_actions = ""

            # csv_line += "---\n"
            # Vector one hot de 41 a 45 posicions fixes on cada posicio indica una carta. 0 = no s'ha jugat la carta, 1 = s'ha jugat la carta
            for action in self.actions:
                # binary_actions += str(action)

                # if self.__model_type != 8:
                inputs_array += [action]

            # if self.__model_type == 8:
            #     total_actions = 2**41 - 1 if self.__is_brisca else 2**45 - 1
            #     inputs_array.append(self.__min_max_normalize(int(binary_actions, 2), 0, total_actions))

        # print(inputs_array)
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

    def heuristics(self, played_card: Card, playable_hand: List[Card]) -> None:
        heuristics: Player_state_heuristic = Player_state_heuristic(played_card, playable_hand, self.__round_played_cards, self.__player_id, self.__num_players, self.__trump, self.all_played_cards, self.__single_mode, self.__sing_declarations, self.__is_brisca)
        self.__heuristics += heuristics.heuristics()
