from tkinter import *
from typing import List, Optional

from constants import WINDOW_X_SIZE, WINDOW_Y_SIZE, PLAYABLE_FRAME_POS_Y, \
    PLAYABLE_FRAME_POS_X, PLAYABLE_FRAME_SIZE_Y, PLAYABLE_FRAME_SIZE_X, BACKGROUND_COLOR, PLAYER_SCORE_SIZE_X, \
    PLAYER_SCORE_SIZE_Y, PLAYER_0_SCORE_POS_X, PLAYER_0_SCORE_POS_Y, PLAYER_1_SCORE_POS_X, PLAYER_1_SCORE_POS_Y, \
    PLAYER_2_SCORE_POS_X, PLAYER_3_SCORE_POS_X, PLAYER_3_SCORE_POS_Y, PLAYER_2_SCORE_POS_Y, \
    INFO_FRAME_SIZE_Y, INFO_FRAME_SIZE_X, INFO_FRAME_POS_Y, INFO_FRAME_POS_X
from playable.playable_info import Playable_info
from playable.playable_score import Playable_score
from playable.playable_utility_functions import raise_frame_or_label


class Playable_frames:
    """
    Classe Playable_frames
    """
    def __init__(self, root: Tk, num_players: int) -> None:
        self.__root: Tk = root
        self.__num_players: int = num_players

        self.__player_scores: List[Optional[Playable_score]] = [None] * num_players

        self.frm_board: Frame = Frame(self.__root, width=WINDOW_X_SIZE, height=WINDOW_Y_SIZE, bg=BACKGROUND_COLOR)
        self.frm_board.pack_propagate(False)
        self.frm_board.pack(padx=0, pady=0)

        # Així creo 4 frames molt petits que envolten la zona on tenia el frame, per tant, les cartes passen "menys" cops per sobre i no deixen gairebe res de rastre
        frm_cards_1: Frame = Frame(self.frm_board, width=PLAYABLE_FRAME_SIZE_X, height=2, bg='green', highlightbackground="black", highlightthickness=2)
        frm_cards_2: Frame = Frame(self.frm_board, width=2, height=PLAYABLE_FRAME_SIZE_Y, bg='green', highlightbackground="black", highlightthickness=2)
        frm_cards_3: Frame = Frame(self.frm_board, width=PLAYABLE_FRAME_SIZE_X, height=2, bg='green', highlightbackground="black", highlightthickness=2)
        frm_cards_4: Frame = Frame(self.frm_board, width=2, height=PLAYABLE_FRAME_SIZE_Y, bg='green', highlightbackground="black", highlightthickness=2)

        frm_cards_1.place(x=PLAYABLE_FRAME_POS_X, y=PLAYABLE_FRAME_POS_Y)
        frm_cards_2.place(x=PLAYABLE_FRAME_POS_X, y=PLAYABLE_FRAME_POS_Y)
        frm_cards_3.place(x=PLAYABLE_FRAME_POS_X, y=PLAYABLE_FRAME_POS_Y + PLAYABLE_FRAME_SIZE_Y)
        frm_cards_4.place(x=PLAYABLE_FRAME_POS_X + PLAYABLE_FRAME_SIZE_X, y=PLAYABLE_FRAME_POS_Y)

        self.frm_info: Frame = Frame(self.frm_board, width=INFO_FRAME_SIZE_X, height=INFO_FRAME_SIZE_Y, bg='white', highlightbackground="black", highlightthickness=2)

        self.frm_info.place(x=INFO_FRAME_POS_X, y=INFO_FRAME_POS_Y)

        self.info: Playable_info = Playable_info(self.frm_info)

        self.frm_score_p1: Frame = Frame(self.frm_board, width=PLAYER_SCORE_SIZE_X, height=PLAYER_SCORE_SIZE_Y, bg='white', highlightbackground="black", highlightthickness=2)
        self.frm_score_p1.place(x=PLAYER_0_SCORE_POS_X, y=PLAYER_0_SCORE_POS_Y)
        self.__player_scores[0] = Playable_score(self.frm_score_p1, 0)

        self.frm_score_p2: Frame = Frame(self.frm_board, width=PLAYER_SCORE_SIZE_X, height=PLAYER_SCORE_SIZE_Y, bg='white', highlightbackground="black", highlightthickness=2)
        self.frm_score_p2.place(x=PLAYER_1_SCORE_POS_X, y=PLAYER_1_SCORE_POS_Y)
        self.__player_scores[1] = Playable_score(self.frm_score_p2, 1)

        if self.__num_players > 2:
            self.frm_score_p3: Frame = Frame(self.frm_board, width=PLAYER_SCORE_SIZE_X, height=PLAYER_SCORE_SIZE_Y, bg='white', highlightbackground="black", highlightthickness=2)
            self.frm_score_p3.place(x=PLAYER_2_SCORE_POS_X, y=PLAYER_2_SCORE_POS_Y)
            self.__player_scores[2] = Playable_score(self.frm_score_p3, 2)

            if self.__num_players > 3:
                self.frm_score_p4: Frame = Frame(self.frm_board, width=PLAYER_SCORE_SIZE_X, height=PLAYER_SCORE_SIZE_Y, bg='white', highlightbackground="black", highlightthickness=2)
                self.frm_score_p4.place(x=PLAYER_3_SCORE_POS_X, y=PLAYER_3_SCORE_POS_Y)
                self.__player_scores[3] = Playable_score(self.frm_score_p4, 3)

        raise_frame_or_label(self.frm_board)

        # self.f2 = Frame(self.root, bg='yellow')
        # self.f3 = Frame(self.root, bg='red')
        # self.f4 = Frame(self.root, bg='grey')

        # for frame in (self.frm_board, self.f2, self.f3, self.f4):
        #    frame.place(x=0, y=0, width=WINDOW_X_SIZE, height=WINDOW_Y_SIZE)

    # Player Score Functions
    def player_score_update_score(self, player_id: int, score: int) -> None:
        player_score: Optional[Playable_score] = self.__player_scores[player_id]
        if player_score is not None:
            player_score.update_score(score)

    def player_score_update_total_score(self, player_id: int, total_score: int, wins: int) -> None:
        player_score: Optional[Playable_score] = self.__player_scores[player_id]
        if player_score is not None:
            player_score.update_total_score(total_score, wins)
