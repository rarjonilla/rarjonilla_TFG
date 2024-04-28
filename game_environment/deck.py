import random
from typing import List, Optional

from game_environment.card import Card
from constants import SUITS, CARDS
from game_environment.suit import Suit


class Deck:

    def __init__(self) -> None:
        self.__deck_cards: List[Card] = []
        self.__trump_card: Optional[Card] = None
        # Quan no queden images, el trump_card serà None i hem de seguir jugant sabent quin era el pal del triomf
        self.__trump_suit_id: int = 0

        self.__init_deck_cards()
        # self.__init_deck_cards_test()

    # Getters
    def get_trump_card(self) -> Optional[Card]:
        return self.__trump_card

    def get_trump_suit_id(self) -> int:
        return self.__trump_suit_id

    # Private functions
    def __init_deck_cards(self) -> None:
        # es recorren els pals
        idx_training_position = 0
        for i_suit in SUITS:
            suit = Suit(i_suit, SUITS[i_suit])
            # Per a cada pal, es recorren el total de images del pal
            for key, card in CARDS.items():
                # self.deck_cards.append(Card(i_suit, card.card_num, card.card_value, card.is_king, card.is_knight))
                self.__deck_cards.append(Card(suit, card["card_num"], card["card_value"], card["training_pos"], idx_training_position))
                idx_training_position += 1

        random.shuffle(self.__deck_cards)

    def __init_deck_cards_test(self) -> None:
        suit_or = Suit(1, SUITS[1])
        suit_bastos = Suit(2, SUITS[2])
        suit_espases = Suit(3, SUITS[3])
        suit_copes = Suit(4, SUITS[4])

        # (Nota personal: Llegir del final al inici)

        # La resta de cartes son indiferents i no les afegeixo per les proves
        # L'últim jugador robarà la carta de triomf
        self.__deck_cards.append(Card(suit_bastos, 2, 0))
        self.__deck_cards.append(Card(suit_copes, 2, 0))
        self.__deck_cards.append(Card(suit_espases, 2, 0))

        self.__deck_cards.append(Card(suit_bastos, 11, 3))
        self.__deck_cards.append(Card(suit_or, 2, 0))
        self.__deck_cards.append(Card(suit_espases, 10, 2))
        self.__deck_cards.append(Card(suit_bastos, 11, 3))

        # Del player 0 al 3 (es reparteixen des de l'última línia fins la primera del següent bloc)
        self.__deck_cards.append(Card(suit_bastos, 7, 0))
        self.__deck_cards.append(Card(suit_copes, 7, 0))

        self.__deck_cards.append(Card(suit_espases, 7, 0))
        self.__deck_cards.append(Card(suit_or, 7, 0))

        self.__deck_cards.append(Card(suit_bastos, 7, 0))
        self.__deck_cards.append(Card(suit_copes, 7, 0))

        self.__deck_cards.append(Card(suit_espases, 7, 0))
        self.__deck_cards.append(Card(suit_espases, 11, 3))

        self.__deck_cards.append(Card(suit_bastos, 7, 0))
        self.__deck_cards.append(Card(suit_copes, 7, 0))

        self.__deck_cards.append(Card(suit_espases, 7, 0))
        self.__deck_cards.append(Card(suit_espases, 12, 4))

        self.__deck_cards.append(Card(suit_bastos, 7, 0))
        self.__deck_cards.append(Card(suit_copes, 7, 0))

        self.__deck_cards.append(Card(suit_espases, 7, 0))
        self.__deck_cards.append(Card(suit_or, 7, 0))

        self.__deck_cards.append(Card(suit_bastos, 1, 11))
        self.__deck_cards.append(Card(suit_copes, 6, 0))

        self.__deck_cards.append(Card(suit_espases, 6, 0))
        self.__deck_cards.append(Card(suit_bastos, 11, 3))

        self.__deck_cards.append(Card(suit_bastos, 7, 0))
        self.__deck_cards.append(Card(suit_copes, 5, 0))

        self.__deck_cards.append(Card(suit_espases, 5, 0))
        self.__deck_cards.append(Card(suit_bastos, 1, 11))

        self.__deck_cards.append(Card(suit_or, 3, 10))
        self.__deck_cards.append(Card(suit_espases, 3, 10))

        # Carta de triomf
        self.__deck_cards.append(Card(suit_or, 10, 2))

        self.__deck_cards.append(Card(suit_espases, 3, 10))
        self.__deck_cards.append(Card(suit_espases, 2, 0))

        self.__deck_cards.append(Card(suit_espases, 3, 10))
        self.__deck_cards.append(Card(suit_espases, 1, 11))

        self.__deck_cards.append(Card(suit_espases, 3, 10))
        self.__deck_cards.append(Card(suit_espases, 2, 0))

    # Functions
    def change_trump_card(self, card: Card) -> None:
        self.__trump_card = card

    def extract_card(self) -> Card:
        # Retornem i eliminem l'últim element de la llista
        # Si no en queda cap, s'ha de retornar el triomf
        if self.get_deck_size() > 0:
            return self.__deck_cards.pop()
        else:
            card: Optional[Card] = self.__trump_card
            self.__trump_card = None

            if card is None:
                raise AssertionError("Card is None")

            return card

    def extract_trump_card(self) -> None:
        self.__trump_card = self.__deck_cards.pop()
        self.__trump_suit_id = self.__trump_card.get_suit_id()

    def get_deck_size(self) -> int:
        return len(self.__deck_cards)

    def get_real_deck_size(self) -> int:
        ds: int = self.get_deck_size()

        if self.__trump_card is not None:
            ds += 1

        return ds

    def has_remaining_cards(self) -> bool:
        return len(self.__deck_cards) > 0

    def has_trump(self) -> bool:
        # print(self.__trump_card is not None)
        return self.__trump_card is not None

    def is_high_trump(self) -> bool:
        return self.__trump_card is not None and self.__trump_card.has_value()
