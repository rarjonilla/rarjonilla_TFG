import copy
import csv

from typing import List, Dict, Optional

from game_environment.card import Card
from game_environment.round import Round
from game_environment.score import Score
from training.player_state import Player_state


class Game_state:
    """Classe training player state....."""

    def __init__(self, is_brisca: bool, single_mode: bool, num_cards: int, num_players: int, rules: Dict, model_type: List[int]) -> None:
        self.__is_brisca = is_brisca
        self.__single_mode = single_mode
        self.__num_cards: int = num_cards
        self.__num_players: int = num_players
        self.__trump: Optional[Card] = None
        self.__rules: Dict = rules
        self.__model_type: List[int] = model_type

        self.__deck_size: int = 0

        self.__score: List[int] = [0] * num_players
        self.__round_played_cards: List[Optional[Card]] = []

        self.__sing_declarations: Optional[List[Optional[int]]] = []

        # Play actual de cada jugador
        self.__current_plays: List[Optional[Player_state]] = []

        # Plays de la ronda actual
        # Cada jugador tindrà entre 0 i 3 plays
        self.__round_plays: List[List[Player_state]] = []

        # Plays de la partida
        self.__game_plays: List[List[Player_state]] = []

        # Plays totals de totes les partides
        self.__all_plays: List[Player_state] = []

        for i in range(num_players):
            self.__round_plays.append([])
            self.__game_plays.append([])
            # self.__all_plays.append([])

        self.__viewed_cards: List[List[Card]] = [] * num_players
        self.__all_played_cards: List[int] = [0] * 40

    def get_player_state(self, player_id: int) -> Player_state:
        player_state = self.__current_plays[player_id]
        return player_state

    # Emmagatzemar CSV
    def save_csv(self, csv_filename: str) -> None:
        with open(csv_filename, 'a', newline='') as csv_file:
            # Crear objecte writer
            csv_writer = csv.writer(csv_file)

            for play in self.__all_plays:
                # Tute-> 1 partida = 5 Kb / 1000 partides = 3.4 Mb (8 segons)
                csv_line = play.csv_line()

                # Tute -> 1 partida = 42 Kb / 1000 partides = 38.9 Mb (11 segons)
                # csv_line = play.csv_line_for_play()
                csv_writer.writerow(csv_line.split(','))

    # Afegim l'acció actual a la llista de jugades del jugador en aquesta ronda
    def add_current_to_round(self, player_id: int) -> None:
        cp = self.__current_plays[player_id]
        if cp is None:
            raise AssertionError("Current play is None")
        self.__round_plays[player_id].append(cp)
        # print(cp.csv_line())

    # S'afegeix la puntuació de la ronda a cada jugada (es farà puntuació positiva per al guanyador i negativa per als perdedors)
    # En cas de 4 jugadors en parella, es considerarà puntuació positiva per als 2 jugadors de l'equip
    def finalize_round(self, round_: Round) -> None:
        for player_id, player_plays in enumerate(self.__round_plays):
            for player_play in player_plays:
                if round_.get_round_winner() == player_id or (not self.__single_mode and (player_id % 2 == round_.get_round_winner() % 2)):
                    player_play.set_round_score(round_.get_round_points())
                # else:
                #     player_play.set_round_score(- round_.get_round_points())

                self.__game_plays[player_id].append(player_play)

            self.__round_plays[player_id] = []

    # S'afegeix si ha guanyat la partida
    def finalize_game(self, score: Score) -> None:
        winners: List[int] = score.get_last_winners()

        for player_id, player_plays in enumerate(self.__game_plays):
            for player_play in player_plays:
                if player_id in winners or (not self.__single_mode and (player_id % 2 in winners)):
                    player_play.set_winner()

                # print(player_play.csv_line())

                self.__all_plays.append(player_play)

            self.__game_plays[player_id] = []

        # print("")

    def add_viewed_card(self, player_id: int, viewed_card: Card) -> None:
        if viewed_card not in self.__viewed_cards[player_id]:
            self.__viewed_cards[player_id].append(viewed_card)
        # print("9999999999999999999999999999999")
        # print(self.__viewed_cards[player_id])

    def remove_viewed_card(self, player_id: int, viewed_card: Card) -> None:
        if viewed_card in self.__viewed_cards[player_id]:
            self.__viewed_cards[player_id].remove(viewed_card)

    def add_played_card(self, played_card: Card) -> None:
        self.__round_played_cards.append(played_card)
        # print(self.__all_played_cards)
        # print(played_card.get_training_idx())
        self.__all_played_cards[played_card.get_training_idx()] = 1

    def change_trump_card(self, trump: Card) -> None:
        self.__trump = trump

    def new_game(self) -> None:
        self.__current_plays = []

        self.__score = [0] * self.__num_players
        self.__round_played_cards = []
        self.__viewed_cards = []
        self.__all_played_cards = [0] * 40

        for id_player in range(self.__num_players):
            self.__current_plays.append(None)
            self.__viewed_cards.append([])

    def set_sing_declarations(self, sing_declarations: Optional[List[Optional[int]]]) -> None:
        self.__sing_declarations = sing_declarations

        #         for i in range(self.__num_players):
        #             sd = sing_declarations[i]
        #             if sd is not None:
        #                 suit: Suit = Suit(sd, "")
        #                 t_idx = sd + ((sd - 1) * 10)
        #                 card: Card = Card(suit, 11, 3, 7, t_idx)
        #                 self.add_viewed_card(i, card)
        #
        #                 t_idx = sd + ((sd - 1) * 10)
        #                 card = Card(suit, 12, 4, 8, t_idx)
        #                 self.add_viewed_card(i, card)

        # for id_player in range(self.__num_players):
        #     self.__current_plays[id_player].set_sing_declarations(sing_declarations)

    def set_action(self, player_id: int, action: int, type: int) -> None:
        cp = self.__current_plays[player_id]
        if cp is None:
            raise AssertionError("Current play is None")

        cp.set_action(action, type)

    def new_round(self, first_round: bool, deck_size: int, score: List[int]) -> None:
        self.__deck_size = deck_size
        self.__score = score

        self.__round_played_cards = []

        # for id_player in range(self.__num_players):

    def new_turn(self, player_id: int, hand: List[Card]) -> None:
        score: List[int] = copy.deepcopy(self.__score)
        sing_declarations: Optional[List[Optional[int]]] = copy.deepcopy(self.__sing_declarations)
        if self.__trump is None:
            raise AssertionError("Trump is None")
        self.__current_plays[player_id] = Player_state(player_id, self.__single_mode, self.__is_brisca, self.__rules, self.__num_cards, self.__num_players, self.__deck_size, self.__trump, score, sing_declarations, self.__model_type[player_id])

        hands: List[List[Card]] = copy.deepcopy(self.__viewed_cards)
        all_played_cards: List[int] = copy.deepcopy(self.__all_played_cards)
        hands[player_id] = hand

        cp = self.__current_plays[player_id]
        if cp is None:
            raise AssertionError("Current play is None")

        cp.set_hands(hands)
        cp.set_all_played_cards(all_played_cards)

        round_played_cards: List[Optional[Card]] = copy.deepcopy(self.__round_played_cards)
        cp.set_round_played_cards(round_played_cards)

    def heuristics(self, player_id: int, played_card: Card, playable_hand: List[Card]) -> None:
        cp = self.__current_plays[player_id]
        if cp is None:
            raise AssertionError("Current play is None")

        cp.heuristics(played_card, playable_hand)

