from typing import List, Dict, Optional
from game_environment.card import Card
from configuration import PRINT_CONSOLE


class Round:
    """
    Classe Round...
    """

    def __init__(self, initial_player: int, trump_suit: int, is_last_round: bool, rules: Dict, sing_declarations: Optional[List[Optional[int]]]) -> None:
        self.__initial_player: int = initial_player
        self.__trump_suit: int = trump_suit
        self.__is_last_round: bool = is_last_round
        self.__rules: Dict = rules
        self.__sing_declarations: Optional[List[Optional[int]]] = sing_declarations
        self.__played_cards: List[Card] = []
        self.__round_winner: int = -1
        self.__round_points: int = 0

        self.__singed_suit: int = 0

    # Getters
    def get_round_points(self) -> int:
        return self.__round_points

    def get_round_winner(self) -> int:
        return self.__round_winner

    def get_sing_declarations(self) -> Optional[List[Optional[int]]]:
        return self.__sing_declarations

    def get_singed_suit(self) -> int:
        return self.__singed_suit

    def has_singed(self) -> bool:
        return self.__singed_suit != 0

    # Functions
    # El retorn només té significat per al Tute, True és que ha cantat i False que no
    # Per a la brisca sempre serà False
    def calc_round_results(self) -> bool:
        if self.__is_last_round and self.__rules['last_tens']:
            self.__round_points += 10

        # Calc winner
        # Es recorren les images en ordre i es comprova si la carta següent guanya a la guanyadora fins al moment
        winner_card: Optional[Card] = None
        round_winner: int = self.__initial_player
        has_used_tute_card: bool = False
        for card_idx, card in enumerate(self.__played_cards):
            # print(card_idx)

            if winner_card is None:
                winner_card = card
                # Només per propòsits de cantar al tute

                sing_dec: Optional[List[Optional[int]]] = self.__sing_declarations
                if sing_dec is not None:
                    sd: Optional[int] = sing_dec[self.__initial_player]
                    if sd is not None:
                        has_used_tute_card = True if self.__sing_declarations is None or (card.is_king() or card.is_knight()) and card.is_same_suit(sd) else False
            elif card.wins_card(winner_card, self.__trump_suit):
                winner_card = card
                round_winner = self.__initial_player + card_idx
                if PRINT_CONSOLE:
                    print("change winner: ", round_winner)

                # Només per propòsits de cantar al tute

                sing_dec = self.__sing_declarations
                if sing_dec is not None:
                    sd = sing_dec[round_winner % len(self.__played_cards)]
                    if sd is not None:
                        has_used_tute_card = True if self.__sing_declarations is None or (card.is_king() or card.is_knight()) and card.is_same_suit(sd) else False

        self.__round_winner = round_winner % len(self.__played_cards)
        if PRINT_CONSOLE:
            print("final winner: ", self.__round_winner)

        if self.__sing_declarations is not None and self.__sing_declarations[self.__round_winner] is not None and not has_used_tute_card:
            if PRINT_CONSOLE:
                print("singed", self.__sing_declarations, self.__sing_declarations[self.__round_winner], has_used_tute_card)

            self.__round_points += 40 if self.is_sing_suit() else 20

            ss: Optional[int] = self.__sing_declarations[self.__round_winner]
            self.__singed_suit = 0 if ss is None else ss

            return True
        elif self.__sing_declarations is not None:
            if PRINT_CONSOLE:
                print("not singed", self.__sing_declarations, self.__sing_declarations[self.__round_winner],
                      has_used_tute_card)

        return False

    def calc_round_results_tute(self, tute_pairs: List[Optional[int]]) -> bool:
        self.calc_round_results()

        if tute_pairs[self.__round_winner] is not None:
            self.__round_points += 40 if tute_pairs[self.__round_winner] == self.__trump_suit else 20
            return True

        return False

    def highest_suit_card(self) -> Optional[Card]:
        # print(self.__played_cards)
        if len(self.__played_cards) == 0:
            return None

        highest_card: Card = self.__played_cards[0]

        for card in self.__played_cards:
            if card.is_same_suit(highest_card.get_suit_id()) and (card.has_higher_value(highest_card.get_value()) or (card.has_same_value(highest_card.get_value()) and card.is_higher_label(highest_card.get_label()))):
                highest_card = card

        return highest_card

    def highest_trump_played(self) -> Optional[Card]:
        if len(self.__played_cards) == 0:
            return None

        highest_trump_card: Optional[Card] = None

        for card in self.__played_cards:
            if card.is_same_suit(self.__trump_suit) and highest_trump_card is None:
                highest_trump_card = card
            elif highest_trump_card is not None and card.is_same_suit(self.__trump_suit) and (card.has_higher_value(highest_trump_card.get_value()) or (card.has_same_value(highest_trump_card.get_value()) and card.is_higher_label(highest_trump_card.get_label()))):
                highest_trump_card = card

        return highest_trump_card

    def hunt_the_three(self, is_last_round: bool) -> int:
        player_hunting: int = -1
        if is_last_round and self.__played_cards[0].is_as() and self.__played_cards[1].is_three() and self.__played_cards[0].is_same_suit(self.__played_cards[1].get_suit_id()) and self.__played_cards[0].is_same_suit(self.__trump_suit):
            player_hunting = self.__initial_player
        elif self.__played_cards[0].is_three() and self.__played_cards[1].is_as() and self.__played_cards[0].is_same_suit(self.__played_cards[1].get_suit_id()) and self.__played_cards[0].is_same_suit(self.__trump_suit):
            player_hunting = self.__initial_player + 1

        return player_hunting % len(self.__played_cards) if player_hunting != -1 else player_hunting

    def is_sing_suit(self) -> bool:
        if self.__sing_declarations is not None:
            return self.__sing_declarations[self.__round_winner] == self.__trump_suit

        return False

    def played_card(self, card: Card) -> None:
        self.__played_cards.append(card)
        self.__round_points += card.get_value()

    # Print
    def show_round(self) -> None:
        player_id: int = self.__initial_player
        for card_idx, card in enumerate(self.__played_cards):
            print("player ", player_id, " played ", card)
            player_id += 1
            player_id %= len(self.__played_cards)

        print("round winner: ", self.__round_winner, " amb ", self.__round_points, " punts")






