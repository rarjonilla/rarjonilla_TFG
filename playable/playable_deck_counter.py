from tkinter import *

from constants import DECK_COUNTER_SIZE_X, DECK_COUNTER_SIZE_Y, DECK_COUNTER_POS_X, DECK_COUNTER_POS_Y


class Playable_deck_counter:
    """
    Classe Playable_deck_counter
    """
    def __init__(self, master_frm: Frame, cards_in_deck: int) -> None:
        self.master_frm: Frame = master_frm
        self.cards_in_deck: int = cards_in_deck

        self.lbl: Label = Label(self.master_frm, text=self.cards_in_deck, width=DECK_COUNTER_SIZE_X, height=DECK_COUNTER_SIZE_Y, borderwidth=2, relief="sunken")
        self.lbl.place(x=DECK_COUNTER_POS_X, y=DECK_COUNTER_POS_Y)

    def update_counter(self, cards_in_deck: int) -> None:
        self.cards_in_deck = cards_in_deck
        self.lbl.config(text=self.cards_in_deck)
