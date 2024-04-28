from tkinter import Tk, Frame, Label

from PIL import Image, ImageTk
from typing import Tuple, Optional

from PIL.ImageTk import PhotoImage

from constants import CARD_X_SIZE, CARD_Y_SIZE, TIME_MOVING_CARD, DEAL_INSTANT_SLEEP_TIME
from configuration import INSTANT_DEAL
from playable.playable_utility_functions import resize_cards, tksleep


class Playable_card:
    """
    Classe Playable_card
    """
    # TODO no cal card_position
    def __init__(self, root: Tk, master_frm: Frame, path_img: str, card_position: Optional[int], hand_pos_x: Optional[int], hand_pos_y: Optional[int], real_pos_x: int, real_pos_y: int, vertical_orientation: bool) -> None:
        self.__root: Tk = root
        self.__master_frm: Frame = master_frm

        # self.card_position: Optional[int] = card_position

        self.__hand_pos_x: Optional[int] = hand_pos_x
        self.__hand_pos_y: Optional[int] = hand_pos_y
        self.__real_pos_x: int = real_pos_x
        self.__real_pos_y: int = real_pos_y

        card_size_x, card_size_y, rotation = self.__calc_card_size(vertical_orientation)

        self.__render_img: PhotoImage = resize_cards(path_img, rotation)
        self.__lbl: Label = Label(self.__master_frm, image=self.__render_img, width=card_size_x, height=card_size_y)
        self.__lbl.place(x=self.__real_pos_x, y=self.__real_pos_y)

        # TEST per les transparències de les imatges png (amb lbl no hi ha transparència)
        # self.lbl = Canvas(master_frm, bd=0, width=card_size_x, height=card_size_y, highlightbackground="black", highlightthickness=0, background=root["bg"])
        # self.lbl.place(x=self.real_pos_x, y=self.real_pos_y)
        # self.lbl.create_image(0, 0, image=self.render_img)

    def get_label(self) -> Label:
        return self.__lbl

    def get_hand_pos_x_not_none(self) -> int:
        if self.__hand_pos_x is None:
            raise AssertionError("hand pos x is None")

        return self.__hand_pos_x

    def get_hand_pos_y_not_none(self) -> int:
        if self.__hand_pos_y is None:
            raise AssertionError("hand pos y is None")

        return self.__hand_pos_y

    # TODO crec que no fa falta i es pot eliminar
    def create_label(self, path_img: str, vertical_orientation: bool, real_pos_x: int, real_pos_y: int) -> None:
        self.__real_pos_x = real_pos_x
        self.__real_pos_y = real_pos_y

        card_size_x, card_size_y, rotation = self.__calc_card_size(vertical_orientation)
        self.__render_img = resize_cards(path_img, rotation)
        self.__lbl = Label(self.__master_frm, image=self.__render_img, width=card_size_x, height=card_size_y)
        self.__lbl.place(x=self.__real_pos_x, y=self.__real_pos_y)

    def destroy_label(self) -> None:
        self.__lbl.destroy()

    def __calc_card_size(self, vertical_orientation: bool) -> Tuple[int, int, int]:
        card_size_x = CARD_X_SIZE if vertical_orientation else CARD_Y_SIZE
        card_size_y = CARD_Y_SIZE if vertical_orientation else CARD_X_SIZE
        rotation = 0 if vertical_orientation else 270

        return card_size_x, card_size_y, rotation

    def clear_label_image(self) -> None:
        self.__lbl.config(image='')

    def change_render(self, new_path_img: str, vertical_orientation: bool) -> None:
        card_size_x, card_size_y, rotation = self.__calc_card_size(vertical_orientation)

        self.__render_img = resize_cards(new_path_img, rotation)
        self.__lbl.config(image=self.__render_img, width=card_size_x, height=card_size_y)

    def show_important_card(self, path_to_img: str, vertical_position: bool, player_id: int) -> None:
        self.change_render(path_to_img, vertical_position)
        adding = 50
        if player_id == 0 or player_id == 3:
            adding = -50

        pos_x = self.__real_pos_x + adding if not vertical_position else self.__real_pos_x
        pos_y = self.__real_pos_y + adding if vertical_position else self.__real_pos_y
        self.move_to_pos(pos_x, pos_y, smoothly=True)

    def hide_important_card(self, path_to_img: str, vertical_position: bool, player_id: int) -> None:
        self.change_render(path_to_img, vertical_position)
        adding = -50
        if player_id == 0 or player_id == 3:
            adding = 50

        pos_x = self.__real_pos_x + adding if not vertical_position else self.__real_pos_x
        pos_y = self.__real_pos_y + adding if vertical_position else self.__real_pos_y
        self.move_to_pos(pos_x, pos_y, smoothly=True)

    def add_playable_border(self) -> None:
        self.__lbl.config(highlightbackground="orange", highlightthickness=3)

    def remove_playable_border(self) -> None:
        self.__lbl.config(highlightthickness=0)

    def change_place(self, new_pos_x: int, new_pos_y: int) -> None:
        # print("changing place from ", self.real_pos_x, self.real_pos_y, " to ", new_pos_x, new_pos_y)
        self.__real_pos_x = new_pos_x
        self.__real_pos_y = new_pos_y
        self.__lbl.place(x=new_pos_x, y=new_pos_y)

    def change_hand_position(self, new_pos_x: int, new_pos_y: int) -> None:
        self.__hand_pos_x = new_pos_x
        self.__hand_pos_y = new_pos_y
        self.move_to_hand()

    def move_to_pos_rotate(self, new_path_img: str, vertical_orientation: bool) -> None:
        self.change_render(new_path_img, vertical_orientation)
        # self.move_to_pos(to_pos_x, to_pos_y)
        self.move_to_hand()

    # TODO -> unificar la funcio move_to_pos i move_to_hand
    def move_to_pos(self, pos_x: int, pos_y: int, smoothly: bool = False) -> None:
        if not INSTANT_DEAL or smoothly:
            another_move = True

            if self.__real_pos_x < pos_x:
                # print("1")
                self.__real_pos_x += 1
            elif self.__real_pos_x > pos_x:
                # print("2")
                self.__real_pos_x -= 1

            if self.__real_pos_y < pos_y:
                # print("3")
                self.__real_pos_y += 1
            elif self.__real_pos_y > pos_y:
                # print("4")
                self.__real_pos_y -= 1

            if self.__real_pos_x == pos_x and self.__real_pos_y == pos_y:
                another_move = False
                # No s'ha fet cap moviment, no cal tornar a cridar a la funció

            self.change_place(self.__real_pos_x, self.__real_pos_y)

            if another_move:
                self.__root.after(TIME_MOVING_CARD, self.move_to_pos, pos_x, pos_y, smoothly)  # in 2 milliseconds, call this function again
        else:
            self.change_place(pos_x, pos_y)
            tksleep(self.__root, DEAL_INSTANT_SLEEP_TIME)

    def move_to_hand(self) -> None:
        if not INSTANT_DEAL:
            if self.__hand_pos_x is None or self.__hand_pos_y is None:
                raise AssertionError("hand_pos_x or hand_pos_y is None")

            another_move = True
            # print(self.real_pos_x, self.real_pos_y, self.hand_pos_x, self.hand_pos_y)
            new_pos_x = self.__real_pos_x
            new_pos_y = self.__real_pos_y
            # if self.real_pos_x < to_pos_x and self.real_pos_y < to_pos_y:

            if self.__real_pos_x < self.__hand_pos_x:
                # print("c")
                new_pos_x += 1
            elif self.__real_pos_x > self.__hand_pos_x:
                # print("d")
                new_pos_x -= 1

            if self.__real_pos_y < self.__hand_pos_y:
                # print("e")
                new_pos_y += 1
            elif self.__real_pos_y > self.__hand_pos_y:
                # print("f")
                new_pos_y -= 1

            if self.__real_pos_x == new_pos_x and self.__real_pos_y == new_pos_y:
                # print("another move false")
                another_move = False
                # No s'ha fet cap moviment, no cal tornar a cridar a la funció

            if another_move:
                # print("another move true")
                self.change_place(new_pos_x, new_pos_y)
                # self.root.after(TIME_MOVING_CARD, self.move_to_pos, to_pos_x, to_pos_y)  # in 2 milliseconds, call this function again
                self.__root.after(TIME_MOVING_CARD, self.move_to_hand)  # in 2 milliseconds, call this function again
        else:
            if self.__hand_pos_x is None or self.__hand_pos_y is None:
                raise AssertionError("hand pos x or hand pos y is None")

            self.change_place(self.__hand_pos_x, self.__hand_pos_y)
            tksleep(self.__root, DEAL_INSTANT_SLEEP_TIME)
