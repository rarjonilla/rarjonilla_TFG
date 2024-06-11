import copy
import csv

from typing import List, Dict, Optional

from game_environment.card import Card
from game_environment.round import Round
from game_environment.score import Score
from training.player_state import Player_state


class Game_state:
    """Classe training player state"""

    def __init__(self, is_brisca: bool, single_mode: bool, num_cards: int, num_players: int, rules: Dict, model_type: List[int], is_supervised_training: bool = False) -> None:
        # Informació de la partida
        self.__is_brisca = is_brisca
        self.__single_mode = single_mode
        self.__num_cards: int = num_cards
        self.__num_players: int = num_players
        self.__trump: Optional[Card] = None
        self.__rules: Dict = rules
        self.__model_type: List[int] = model_type

        # L'entrenament SL és una mica diferent perquè s'haurà d'emmagatzemer el conjunt de dades
        self.__is_supervised_training = is_supervised_training

        # Cartes restants a la baralla
        self.__deck_size: int = 0
        # Puntuació actual
        self.__score: List[int] = [0] * num_players
        # Cartes jugades a la ronda actual
        self.__round_played_cards: List[Optional[Card]] = []
        # Cants de la ronda
        self.__sing_declarations: Optional[List[Optional[int]]] = []
        # Cartes conegudes dels rivals
        self.__viewed_cards: List[List[Card]] = [] * num_players
        # Cartes jugades al llarg de la partida
        self.__all_played_cards: List[int] = [0] * 40

        # Estat del joc corresponent a cada jugador per al torn actual ()
        self.__current_plays: List[Optional[Player_state]] = []

        # Estats dels jocs de la ronda actual per a cada jugador
        # Cada jugador tindrà entre 0 i 3 plays (doble intercanvi de carta i carta jugada)
        self.__round_plays: List[List[Player_state]] = []

        # Llista que guarda les jugades de tota la partida de la partida
        self.__game_plays: List[List[Player_state]] = []

        # Llista que guarda les jugades de totes les partides
        self.__all_plays: List[Player_state] = []

        # Inicialització de les llistes
        for i in range(num_players):
            self.__round_plays.append([])
            self.__game_plays.append([])

    def get_player_state(self, player_id: int) -> Player_state:
        # Retorna l'estat del joc per al torn del jugador
        player_state = self.__current_plays[player_id]
        return player_state

    # Emmagatzemar CSV
    def save_csv(self, csv_filename: str) -> None:
        with open(csv_filename, 'a', newline='') as csv_file:
            # Crear objecte writer
            csv_writer = csv.writer(csv_file)

            for play in self.__all_plays:
                # Tute -> 1 partida = 5 Kb / 1000 partides = 3.4 Mb (8 segons)
                csv_line = play.csv_line()

                # Tute -> 1 partida = 42 Kb / 1000 partides = 38.9 Mb (11 segons)
                csv_writer.writerow(csv_line.split(','))

    # Afegim l'acció actual a la llista de jugades del jugador en aquesta ronda
    def add_current_to_round(self, player_id: int) -> None:
        cp = self.__current_plays[player_id]
        if cp is None:
            raise AssertionError("Current play is None")
        self.__round_plays[player_id].append(cp)

    # S'afegeix la puntuació de la ronda a cada jugada (es farà puntuació positiva per al guanyador i negativa per als perdedors)
    # En cas de 4 jugadors en parella, es considerarà puntuació positiva per als 2 jugadors de l'equip (si guanya el meu company, també guanyo jo)
    def finalize_round(self, round_: Round) -> None:
        for player_id, player_plays in enumerate(self.__round_plays):
            for player_play in player_plays:
                if round_.get_round_winner() == player_id or (not self.__single_mode and (player_id % 2 == round_.get_round_winner() % 2)):
                    player_play.set_round_score(round_.get_round_points())
                # La puntuació negativa no funciona bé, s'elimina
                # else:
                #     player_play.set_round_score(- round_.get_round_points())

                # Aquesta llista només serveix per emmagatzemar el conjun de la SL
                if self.__is_supervised_training:
                    self.__game_plays[player_id].append(player_play)

            # S'inicialitza la llista de jugades de la ronda
            self.__round_plays[player_id] = []

    # S'afegeix als estats del joc de tota la partida si el jugador ha guanyat la partida o no (SL win or lose)
    def finalize_game(self, score: Score) -> None:
        if self.__is_supervised_training:
            winners: List[int] = score.get_last_winners()

            for player_id, player_plays in enumerate(self.__game_plays):
                for player_play in player_plays:
                    if player_id in winners or (not self.__single_mode and (player_id % 2 in winners)):
                        player_play.set_winner()

                    if self.__is_supervised_training:
                        self.__all_plays.append(player_play)

                self.__game_plays[player_id] = []

    # S'afegeix una carta a la llista de cartes conegudes dels rivals
    def add_viewed_card(self, player_id: int, viewed_card: Card) -> None:
        if viewed_card not in self.__viewed_cards[player_id]:
            self.__viewed_cards[player_id].append(viewed_card)

    # S'elimina una carta a la llista de cartes conegudes dels rivals
    def remove_viewed_card(self, player_id: int, viewed_card: Card) -> None:
        if viewed_card in self.__viewed_cards[player_id]:
            self.__viewed_cards[player_id].remove(viewed_card)

    # S'afegeix una carta jugada
    def add_played_card(self, played_card: Card) -> None:
        self.__round_played_cards.append(played_card)
        self.__all_played_cards[played_card.get_training_idx()] = 1

    # S'ha canviar la carta de triomf
    def change_trump_card(self, trump: Card) -> None:
        self.__trump = trump

    # Nova partida, s'inicialitzen totes les llistes
    def new_game(self) -> None:
        self.__current_plays = []

        self.__score = [0] * self.__num_players
        self.__round_played_cards = []
        self.__viewed_cards = []
        self.__all_played_cards = [0] * 40

        for id_player in range(self.__num_players):
            self.__current_plays.append(None)
            self.__viewed_cards.append([])

    # S'indica els cants dels jugadors
    def set_sing_declarations(self, sing_declarations: Optional[List[Optional[int]]]) -> None:
        self.__sing_declarations = sing_declarations

    # S'afegeix una acció a l'estat del joc
    def set_action(self, player_id: int, action: int, type: int) -> None:
        cp = self.__current_plays[player_id]
        if cp is None:
            raise AssertionError("Current play is None")

        cp.set_action(action, type)

    # Nova ronda, s'actualitza les cartes de la baralla i la puntuació
    def new_round(self, first_round: bool, deck_size: int, score: List[int]) -> None:
        self.__deck_size = deck_size
        self.__score = score

        self.__round_played_cards = []

    # S'inicia un nou torn del jugador (s'indica les cartes que té a la mà)
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

    # Es calcula l'heurístic de la jugada (només per SL)
    def heuristics(self, player_id: int, played_card: Card, playable_hand: List[Card]) -> None:
        cp = self.__current_plays[player_id]
        if cp is None:
            raise AssertionError("Current play is None")

        cp.heuristics(played_card, playable_hand)

