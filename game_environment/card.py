from constants import SUITS
from game_environment.suit import Suit


class Card:
    """Classe carta"""

    def __init__(self, suit: Suit, id_label: int, value: int, training_value: int = 0, training_idx: int = 0) -> None:
        # Coll de la carta
        self.__suit = suit
        # Número de la carta (1, 2, ..., 12)
        self.__label: int = id_label
        # Valor de la carta (0, 2, 3, 4, 10, 11)
        self.__value: int = value

        # Només per training
        # Ordre de preferència (de 1 a 10): 2, 4, 5, 6, 7, 10, 11, 12, 3, As
        self.__training_value: int = training_value
        # identificador de la carta pel vector de l'estat en ordre de preferència i per colls (0-9 ors, 10, 19 bastos, 20 a 29 espases, 30 a 39 copes)
        self.__training_idx: int = training_idx

    # Getters
    def get_label(self) -> int:
        return self.__label

    def get_value(self) -> int:
        return self.__value

    def get_training_value(self) -> int:
        return self.__training_value

    def get_training_idx(self) -> int:
        return self.__training_idx

    #     def get_one_hot(self) -> str:
    #         one_hot = self.__suit.get_one_hot()
    #
    #         for i in range(1, 11):
    #             one_hot += "1, " if i == self.__training_value else "0, "
    #         return one_hot

    # Functions
    # TODO - s'ha de canviar tots els if que comproven si la carta té més valor o si te el mateix valor però més label que una altra per aquesta funció
    def has_more_preference(self, card_to_compare: 'Card') -> bool:
        # Retorna True si la carta té més preferència que la que arriba per paràmetre
        return self.is_same_suit(card_to_compare.get_suit_id()) and (self.has_higher_value(card_to_compare.get_value()) or (self.has_same_value(card_to_compare.get_value()) and self.is_higher_label(card_to_compare.get_label())))

    def has_higher_value(self, value: int) -> bool:
        return self.__value > value

    def has_same_training_idx(self, training_idx: int) -> bool:
        return self.__training_idx == training_idx

    def has_same_value(self, value: int) -> bool:
        return self.__value == value

    def has_value(self) -> bool:
        return self.__value > 0

    def is_as(self) -> bool:
        return self.__label == 1

    def is_higher_label(self, label: int) -> bool:
        return self.__label > label

    def is_king(self) -> bool:
        return self.__label == 12

    def is_knight(self) -> bool:
        return self.__label == 11

    def is_seven(self) -> bool:
        return self.__label == 7

    def is_three(self) -> bool:
        return self.__label == 3

    def is_two(self) -> bool:
        return self.__label == 2

    def wins_card(self, winner_card: 'Card', trump_suit_id: int) -> bool:
        # Mateix pal, cal comprovar quina carta té més valor
        if self.is_same_suit(winner_card.get_suit_id()):
            if self.__value < winner_card.__value or (self.__value == winner_card.__value and self.__label < winner_card.__label):
                return False
            else:
                return True
        # Diferent pal, si la carta és del pal del triomf, guanya, sinó, perd
        elif self.is_same_suit(trump_suit_id):
            return True
        else:
            return False

    # Suit Getters
    def get_suit_id(self) -> int:
        return self.__suit.get_id()

    def get_suit_label(self) -> int:
        return self.__suit.get_label()

    def is_same_suit(self, suid_id: int) -> bool:
        return self.__suit.is_same_suit(suid_id)

    # Print
    def __str__(self) -> str:
        return str(self.__label) + ' de ' + str(self.__suit.get_label()) + " -- Value = " + str(self.__value)


