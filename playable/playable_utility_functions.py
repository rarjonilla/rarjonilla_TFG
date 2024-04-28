from __future__ import annotations

from tkinter import Tk, Frame, Label
from typing import Tuple, Optional
from PIL import Image, ImageTk

from game_environment.card import Card
from constants import HORIZONTAL_PLAYER_SIZE_X, CARD_X_SIZE, VERTICAL_PLAYER_SIZE_Y, HORIZONTAL_PLAYER_POS_X, \
    VERTICAL_PLAYER_POS_Y, PLAYER_0_POS_Y, PLAYER_1_POS_X, PLAYER_2_POS_Y, PLAYER_3_POS_X, PLAYABLE_CARD_INITIAL_POS_Y, \
    PLAYABLE_CARD_INITIAL_POS_X_2_PLAYERS, PLAYABLE_CARD_INITIAL_X_WIDTH_2_PLAYERS, \
    PLAYABLE_CARD_INITIAL_POS_X_3_PLAYERS, PLAYABLE_CARD_INITIAL_X_WIDTH_3_PLAYERS, \
    PLAYABLE_CARD_INITIAL_POS_X_4_PLAYERS, PLAYABLE_CARD_INITIAL_X_WIDTH_4_PLAYERS


def calc_dragged_card_position(card_pos_x: int, player_mod_2: bool, num_cards: int) -> int:
    internal_padding = calc_internal_padding(num_cards, player_mod_2)
    # Formula original per jugador 1, ara busquem card_pos
    # card_pos_x = (card_pos * CARD_X_SIZE) + (internal_padding * (card_pos + 1)) + HORIZONTAL_PLAYER_POS_X
    # card_pos_x - HORIZONTAL_PLAYER_POS_X = (card_pos * CARD_X_SIZE) + (internal_padding * (card_pos + 1))
    # card_pos_x - HORIZONTAL_PLAYER_POS_X = card_pos * CARD_X_SIZE + internal_padding * card_pos + internal_padding
    # card_pos_x - HORIZONTAL_PLAYER_POS_X - internal_padding = card_pos * CARD_X_SIZE + internal_padding * card_pos
    # card_pos_x - HORIZONTAL_PLAYER_POS_X - internal_padding = card_pos * (CARD_X_SIZE + internal_padding)
    # (card_pos_x - HORIZONTAL_PLAYER_POS_X - internal_padding) / (CARD_X_SIZE + internal_padding) = card_pos
    card_pos = round((card_pos_x - HORIZONTAL_PLAYER_POS_X - internal_padding) / (CARD_X_SIZE + internal_padding))
    return card_pos


def calc_hand_card_position(player_id: int, card_pos: int, num_cards: int) -> Tuple[int, int]:
    card_pos_x = 0
    card_pos_y = 0

    internal_padding = calc_internal_padding(num_cards, player_id % 2 == 0)

    if player_id % 2 == 0:
        card_pos_x = (card_pos * CARD_X_SIZE) + (internal_padding * (card_pos + 1)) + HORIZONTAL_PLAYER_POS_X
    else:
        card_pos_y = (card_pos * CARD_X_SIZE) + (internal_padding * (card_pos + 1)) + VERTICAL_PLAYER_POS_Y

    if player_id == 0:
        card_pos_y = PLAYER_0_POS_Y
    elif player_id == 1:
        card_pos_x = PLAYER_1_POS_X
    elif player_id == 2:
        card_pos_y = PLAYER_2_POS_Y
    elif player_id == 3:
        card_pos_x = PLAYER_3_POS_X

    return card_pos_x, card_pos_y


def calc_internal_padding(num_cards: int, player_mod_2: bool) -> int:
    # external_padding = 10

    if player_mod_2:
        total_size = HORIZONTAL_PLAYER_SIZE_X
        # return (total_size - 2 * external_padding - self.game_environment.num_cards * CARD_X_SIZE) // 4
        return (total_size - num_cards * CARD_X_SIZE) // (num_cards + 1)
    else:
        total_size = VERTICAL_PLAYER_SIZE_Y
        # return (total_size - 2 * external_padding - self.game_environment.num_cards * CARD_X_SIZE) // 4
        return (total_size - num_cards * CARD_X_SIZE) // (num_cards + 1)


def calc_played_card_position(num_players: int, cards_played: int) -> Tuple[int, int]:
    x_initial = 0
    y_initial = PLAYABLE_CARD_INITIAL_POS_Y
    x_width = 0
    if num_players == 2:
        x_initial = PLAYABLE_CARD_INITIAL_POS_X_2_PLAYERS
        x_width = PLAYABLE_CARD_INITIAL_X_WIDTH_2_PLAYERS
    elif num_players == 3:
        x_initial = PLAYABLE_CARD_INITIAL_POS_X_3_PLAYERS
        x_width = PLAYABLE_CARD_INITIAL_X_WIDTH_3_PLAYERS
    elif num_players == 4:
        x_initial = PLAYABLE_CARD_INITIAL_POS_X_4_PLAYERS
        x_width = PLAYABLE_CARD_INITIAL_X_WIDTH_4_PLAYERS

    x = x_initial + cards_played * x_width
    y = y_initial

    return x, y


def get_card_path(card: Optional[Card]) -> str:
    if card is None:
        return "images/backgrounds/background.png"
    else:
        return f'images/cards/{card.get_label()}_de_{card.get_suit_label()}.png'


def raise_frame_or_label(element: Frame | Label) -> None:
    element.tkraise()


def resize_cards(path: str, rotate: int) -> ImageTk.PhotoImage:
    card_img = Image.open(path)
    # print(card_img.mode)
    # card_resize_img = card_img.resize((CARD_X_SIZE, CARD_Y_SIZE))
    # print(card_resize_img.mode)

    if rotate > 0:
        card_img = card_img.rotate(rotate, Image.NEAREST, expand=True)

    # global card
    # card = ImageTk.PhotoImage(card_resize_img)
    card = ImageTk.PhotoImage(card_img)

    return card


def tksleep(root: Tk, time: float) -> None:
    """
    Emulating `time.sleep(seconds)`
    Created by TheLizzard, inspired by Thingamabobs
    """
    root.after(int(time * 1000), root.quit)
    root.mainloop()
