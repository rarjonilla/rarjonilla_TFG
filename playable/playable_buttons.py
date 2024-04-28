from tkinter import *
from typing import Callable

from constants import BTN_CHANGE_POS_X, BTN_CHANGE_POS_Y, BTN_SING_SWORDS_POS_Y, BTN_SING_SWORDS_POS_X, BTN_SING_CUPS_POS_Y, BTN_SING_CUPS_POS_X, \
    BTN_SING_COARSE_POS_Y, BTN_SING_COARSE_POS_X, BTN_SING_GOLD_POS_Y, BTN_SING_GOLD_POS_X, BTN_SIZE_X, BTN_NEW_GAME_POS_X, BTN_NEW_GAME_POS_Y
from playable.playable_button import Playable_button


class Playable_buttons:
    """
    Classe Playable_buttons
    """

    def __init__(self, master_frm: Frame, is_brisca: bool, can_change_rule: bool, func_new_game: Callable, func_choose_change_card: Callable, func_choose_sing_suit: Callable, human_player) -> None:
        self.__human_player = human_player
        self.__new_game: Callable = func_new_game
        self.__choose_change_card: Callable = func_choose_change_card
        self.__choose_sing_suit: Callable = func_choose_sing_suit

        self.__master_frm: Frame = master_frm
        self.__is_brisca: bool = is_brisca
        self.__can_change_rule: bool = can_change_rule

        self.btn_new_game: Playable_button = Playable_button(self.__master_frm, "Nova partida", False, BTN_SIZE_X, BTN_NEW_GAME_POS_X, BTN_NEW_GAME_POS_Y, RAISED, self.__new_game)

        if self.__can_change_rule and self.__human_player:
            self.btn_change_card: Playable_button = Playable_button(self.__master_frm, "Intercanviar carta", False, BTN_SIZE_X, BTN_CHANGE_POS_X, BTN_CHANGE_POS_Y, RAISED, self.__choose_change_card)

        if not self.__is_brisca and self.__human_player:
            self.btn_sing_gold: Playable_button = Playable_button(self.__master_frm, "Cantar en ors", False, BTN_SIZE_X, BTN_SING_GOLD_POS_X, BTN_SING_GOLD_POS_Y, RAISED, lambda: self.__choose_sing_suit(1))
            self.btn_sing_coarse: Playable_button = Playable_button(self.__master_frm, "Cantar en bastos", False, BTN_SIZE_X, BTN_SING_COARSE_POS_X, BTN_SING_COARSE_POS_Y, RAISED, lambda: self.__choose_sing_suit(2))
            self.btn_sing_swords: Playable_button = Playable_button(self.__master_frm, "Cantar en espases ", False, BTN_SIZE_X, BTN_SING_SWORDS_POS_X, BTN_SING_SWORDS_POS_Y, RAISED, lambda: self.__choose_sing_suit(3))
            self.btn_sing_cups: Playable_button = Playable_button(self.__master_frm, "Cantar en copes", False, BTN_SIZE_X, BTN_SING_CUPS_POS_X, BTN_SING_CUPS_POS_Y, RAISED, lambda: self.__choose_sing_suit(4))

    def disable_buttons(self) -> None:
        if not self.__is_brisca and self.__human_player:
            self.btn_sing_gold.disable_button()
            self.btn_sing_coarse.disable_button()
            self.btn_sing_swords.disable_button()
            self.btn_sing_cups.disable_button()
