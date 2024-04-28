import copy
from typing import List, Tuple, Optional
from game_environment.card import Card
from configuration import PRINT_CONSOLE


class Hand:
    """
    Classe Hand...
    """
    def __init__(self, only_assist_rule: bool) -> None:
        self.__cards: List[Card] = []
        self.__singed_suits: List[int] = []

        self.__only_assist_rule: bool = only_assist_rule

    # Getters
    def get_cards_copy(self) -> List[Card]:
        return copy.deepcopy(self.__cards)

    def get_singed_suits(self) -> List[int]:
        return self.__singed_suits

    # Functions
    def add_card(self, card: Card) -> None:
        self.__cards.append(card)

    def add_singed_tute_suit(self, suit_id: int) -> None:
        self.__singed_suits.append(suit_id)

    def black_hand_cards_position(self, trump_suit_id: int) -> Optional[List[int]]:
        card_positions: List[int] = []
        for i, card in enumerate(self.__cards):
            if (card.is_as() or card.is_king() or card.is_three()) and card.is_same_suit(trump_suit_id):
                card_positions.append(i)

        if len(card_positions) == 3:
            return card_positions

        return None

    def can_change(self, change_card_is_higher_than_seven: bool, round_suit_id: int) -> bool:
        # print("can_change")
        # print("change_card_is_higher_than_seven: " + str(change_card_is_higher_than_seven))
        # print("round_suit: " + SUITS[round_suit_id])

        for card in self.__cards:
            if change_card_is_higher_than_seven and card.is_same_suit(round_suit_id) and card.is_seven():
                # print("te la carta per canviar")
                # print(card)
                return True
            elif not change_card_is_higher_than_seven and card.is_same_suit(round_suit_id) and card.is_two():
                # print("te la carta per canviar")
                # print(card)
                return True

        return False

    def cards_of_suit(self, suit_id: int) -> List[Card]:
        suit_cards: List[Card] = []

        for card in self.__cards:
            # if card.get_suit_id() == suit_id:
            if card.is_same_suit(suit_id):
                suit_cards.append(card)

        return suit_cards

    def card_to_change(self, change_card_is_higher_than_seven: bool, round_suit_id: int) -> Tuple[int, Optional[Card]]:
        for i, card in enumerate(self.__cards):
            if (change_card_is_higher_than_seven and card.is_same_suit(round_suit_id) and card.is_seven()) or (not change_card_is_higher_than_seven and card.get_suit_id() == round_suit_id and card.is_two()):
                return i, card

        return 0, None

    def cards_in_hand(self) -> int:
        return len(self.__cards)

    def change_card(self, new_card: Card, card_position: int) -> None:
        self.__cards[card_position] = new_card

    def get_card_in_position(self, card_position: int) -> Tuple[int, Card]:
        card = self.__cards[card_position]
        self.remove_card(card)

        return card_position, card

    def get_card_in_position_no_remove(self, card_position: int) -> Tuple[int, Card]:
        card = self.__cards[card_position]
        return card_position, card

    def get_card_position(self, card: Card) -> int:
        if PRINT_CONSOLE:
            print(self.__cards)
        if card in self.__cards:
            if PRINT_CONSOLE:
                print("if")
            return self.__cards.index(card)

        if PRINT_CONSOLE:
            print("else")
        return -1

    def get_playable_cards_positions(self, trump_suit_id: int, highest_suit_card: Optional[Card], deck_has_cards: bool, highest_trump_played: Optional[Card]) -> List[int]:
        # Brisca
        # Pot triar la carta de la seva mà que vulgui

        # Tute -> only_assist = False
        # https://es.wikihow.com/jugar-al-tute
        # https://es.wikipedia.org/wiki/Tute

        # Si és la primera acció (juga la primera carta d'aquesta ronda), pot triar la carta que vulgui

        # Si no és la primera acció (ja hi ha alguna carta en joc), ha de triar una carta del mateix pal
        #   A més, si disposa d'una carta del mateix pal, però superior, està obligat a tirar-la (no pot tirar una de més baixa si en té una de més alta)
        #   En cas que no tingui cap carta del mateix pal, està obligat a tirar una carta del pal del triomf
        #   Si algú ja ha tirat un triomf perque no te del mateix pal, el jugador ha de seguir tirant una carta del mateix pal inicial, però no cal que sigui superior a les altres
        #   Si algú ja ha tirat un triomf perque no te del mateix pal, i el jugador no té cap carta del pal, si el jugador té una carta de triomf superior a la jugada, ha de tirar-la
        #   Si algú ja ha tirat un triomf perque no te del mateix pal, i el jugador no té cap carta del pal, si el jugador té una carta de triomf, però no superior a la jugada, pot tirar la carta que vulgui
        #   Si no té carta del mateix pal ni de triomf, pot tirar la carta que vulgui

        # Tute -> only_assist = True
        # Els jugadors només tenen l'obligació d'assistir (no cal montar, fallar ni trepitjar)

        # trump_suit_id: int, first_round_card: Card | None, deck_has_cards: bool, round_trump_played: bool
        playable_cards: List[int] = []
        all_cards: List[int] = []
        same_trump_suit_cards: List[int] = []
        same_trump_suit_cards_higher: List[int] = []
        same_played_suit_cards: List[int] = []
        same_played_suit_cards_higher: List[int] = []

        # print("trump_suit_id", trump_suit_id)
        # print("highest_suit_card", highest_suit_card)
        # print("highest_trump_played", highest_trump_played)

        # Recopilacio de les diferents combinacions possibles
        for i, card in enumerate(self.__cards):
            all_cards.append(i)

            if highest_suit_card is not None and card.is_same_suit(highest_suit_card.get_suit_id()):
                same_played_suit_cards.append(i)
                if card.has_higher_value(highest_suit_card.get_value()) or (card.has_same_value(highest_suit_card.get_value()) and card.is_higher_label(highest_suit_card.get_label())):
                    same_played_suit_cards_higher.append(i)

            if card.is_same_suit(trump_suit_id):
                same_trump_suit_cards.append(i)
                if highest_trump_played is not None and card.has_higher_value(highest_trump_played.get_value())  or (highest_trump_played is not None and card.has_same_value(highest_trump_played.get_value()) and card.is_higher_label(highest_trump_played.get_label())):
                    same_trump_suit_cards_higher.append(i)

        # Aplicació de les regles
        # print("trump_suit_id ", trump_suit_id)
        # print("highest_suit_card ", highest_suit_card)
        # print("deck_has_cards ", deck_has_cards)
        # print("highest_trump_played ", highest_trump_played)
        # print("same_played_suit_cards ", same_played_suit_cards)
        # print("same_played_suit_cards_higher ", same_played_suit_cards_higher)
        # print("same_trump_suit_cards ", same_trump_suit_cards)
        # print("same_trump_suit_cards_higher ", same_trump_suit_cards_higher)

        if highest_suit_card is None:
            if PRINT_CONSOLE:
                print("Primera ronda, pot jugar el que vulgui")
            playable_cards = all_cards
        elif not self.__only_assist_rule and highest_trump_played is None and len(same_played_suit_cards_higher) > 0:
            if PRINT_CONSOLE:
                print("Muntar -> No s'ha jugat cap triomf i el jugador té cartes del mateix pal superiors a la millor carta")
            playable_cards = same_played_suit_cards_higher
        elif not self.__only_assist_rule and highest_suit_card.is_same_suit(trump_suit_id) and len(same_played_suit_cards_higher) > 0:
            if PRINT_CONSOLE:
                print("Muntar -> La primera carta és de trioms i el jugador té cartes de triomf superiors a la millor carta")
            playable_cards = same_played_suit_cards_higher
        # elif highest_trump_played is not None and len(same_played_suit_cards) > 0:
        elif len(same_played_suit_cards) > 0:
            if PRINT_CONSOLE:
                print("Asistir -> S'ha jugat alguna carta de triomf i el jugador té cartes del mateix pal")
            playable_cards = same_played_suit_cards
        elif not self.__only_assist_rule and highest_trump_played is None and len(same_trump_suit_cards) > 0:
            if PRINT_CONSOLE:
                print("Fallar - No s'ha jugat cap carta de triomf i el jugador no té cartes del pal principal però si de triomf")
            playable_cards = same_trump_suit_cards
        elif not self.__only_assist_rule and highest_trump_played is not None and len(same_trump_suit_cards_higher) > 0:
            if PRINT_CONSOLE:
                print("Trepitjar - S'ha jugat alguna carta de triomf, el jugador no té cartes del pal principal i té cartes de triomf superiors")
            playable_cards = same_trump_suit_cards_higher
        else:
            if PRINT_CONSOLE:
                print("Contrafallar - el jugador pot triar la carta que vulgui")
            playable_cards = all_cards

        if PRINT_CONSOLE:
            print("playable cards: ")
            for pl_c in playable_cards:
                print(self.__cards[pl_c])

            print("")

        return playable_cards

    def get_playable_cards(self, trump_suit_id: int, highest_suit_card: Optional[Card], deck_has_cards: bool, highest_trump_played: Optional[Card]) -> List[Card]:
        card_pos: List[int] = self.get_playable_cards_positions(trump_suit_id, highest_suit_card, deck_has_cards, highest_trump_played)
        card_list: List[Card] = []

        for cp in card_pos:
            pos, card = self.get_card_in_position_no_remove(cp)
            card_list.append(card)

        return card_list

    def get_sing_cards_position(self, singed_suit_id: int) -> List[int]:
        positions: List[int] = []
        for i, card in enumerate(self.__cards):
            if card.is_same_suit(singed_suit_id) and (card.is_king() or card.is_knight()):
                positions.append(i)

        return positions

    def has_black_hand(self, trump_suit_id: int) -> bool:
        black_hand_cards: int = 0
        for card in self.__cards:
            if (card.is_as() or card.is_king() or card.is_three()) and card.is_same_suit(trump_suit_id):
                black_hand_cards += 1

        return black_hand_cards == 3

    def has_tute(self) -> Tuple[bool, bool]:
        kings: int = 0
        knights: int = 0
        for card in self.__cards:
            if card.is_king():
                kings += 1
            elif card.is_knight():
                knights += 1

        return knights == 4 or kings == 4, kings == 4

    def remove_card(self, card: Card) -> None:
        self.__cards.remove(card)

    def sing_suits_in_hand(self) -> List[int]:
        sing_declarations: List[int] = [0] * 4
        sing_suits_ids: List[int] = []

        for card in self.__cards:
            if (card.is_king() or card.is_knight()) and not card.get_suit_id() in self.__singed_suits:
                sing_declarations[card.get_suit_id() - 1] += 1

            if sing_declarations[card.get_suit_id() - 1] == 2 and card.get_suit_id() not in sing_suits_ids:
                sing_suits_ids.append(card.get_suit_id())

        # Es retorna els suits_ids dels tutes que té a la mà (en té el rey i el cavaller del mateix pal)
        if PRINT_CONSOLE:
            if len(sing_suits_ids) > 0:
                print(len(sing_suits_ids), " tutes a la mà", sing_suits_ids)
        return sing_suits_ids

    def tute_cards_position(self) -> Optional[List[int]]:
        kings_card_positions: List[int] = []
        knights_card_positions: List[int] = []
        for i, card in enumerate(self.__cards):
            if card.is_king():
                kings_card_positions.append(i)
            elif card.is_knight():
                knights_card_positions.append(i)

        if len(kings_card_positions) == 4:
            return kings_card_positions
        elif len(knights_card_positions) == 4:
            return knights_card_positions

        return None

    # Print
    def __str__(self) -> str:
        cards_str = ""
        for card in self.__cards:
            cards_str += str(card) + "\n"
        return cards_str
