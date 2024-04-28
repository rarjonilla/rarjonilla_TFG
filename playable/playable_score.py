from tkinter import *
import tkinter.font as tkFont

from constants import PLAYER_SCORE_LABEL_SIZE_X


class Playable_score:
    """
    Classe Playable_score
    """
    def __init__(self, master_frm: Frame, player_id: int) -> None:
        self.__master_frm: Frame = master_frm
        self.__player_id: int = player_id

        self.__position: str = ""

        if player_id == 0:
            self.__position = " (bottom)"
        elif player_id == 1:
            self.__position = " (left)"
        elif player_id == 2:
            self.__position = " (top)"
        else:
            self.__position = " (right)"

        self.__wins: int = 0
        self.__score: int = 0
        self.__total_score: int = 0

        self.lbl_player: Label = Label(self.__master_frm, text=self.__player_text(), background="white")
        self.lbl_score: Label = Label(self.__master_frm, text=self.__score_text(), width=PLAYER_SCORE_LABEL_SIZE_X, background="white", anchor=NW, justify="left")
        self.__font = tkFont.Font(font=self.lbl_player['font']).actual()

        self.lbl_score.configure(font=(self.__font['family'], 9, 'normal'))
        self.lbl_player.configure(font=(self.__font['family'], 14, 'bold'))

        self.lbl_player.pack(fill=X)
        self.lbl_score.pack(fill=X)

    def __player_text(self) -> str:
        return "Player " + str(self.__player_id) + self.__position

    def __score_text(self) -> str:
        return "Score: " + str(self.__score) + "\nTotal score: " + str(self.__total_score) + " - Wins: " + str(self.__wins)

    def update_score(self, score: int) -> None:
        self.__score = score
        self.lbl_score.config(text=self.__score_text())

    def update_total_score(self, total_score: int, wins: int) -> None:
        self.__total_score = total_score
        self.__wins = wins

        self.lbl_score.config(text=self.__score_text())
