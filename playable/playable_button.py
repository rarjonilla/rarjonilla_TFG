from tkinter import *
from typing import Literal, Callable, Optional


class Playable_button:
    """
    Classe Playable_button
    """

    def __init__(self, master_frm: Frame, text: str, enabled: bool, width: int, pos_x: int, pos_y: int, relief: Literal["raised", "sunken", "flat", "ridge", "solid", "groove"], func: Callable) -> None:
        self.__master_frm: Frame = master_frm

        self.btn: Optional[Button] = None
        if func is not None:
            self.btn = Button(self.__master_frm, text=text, width=width, relief=relief, activeforeground="green", foreground="green", disabledforeground="red", command=func)
        else:
            self.btn = Button(self.__master_frm, text=text, width=width, relief=relief, activeforeground="green", foreground="green", disabledforeground="red")

        self.btn['state'] = DISABLED if not enabled else NORMAL
        self.btn.place(x=pos_x, y=pos_y)

    def disable_button(self) -> None:
        if self.btn is not None:
            self.btn.config(relief=RAISED, disabledforeground="red", highlightbackground="red", highlightthickness=0)
            self.btn['state'] = DISABLED

    def enable_button(self) -> None:
        if self.btn is not None:
            self.btn.config(relief=RAISED, highlightbackground="black", highlightthickness=6)
            self.btn['state'] = NORMAL

    def clicked_button(self) -> None:
        if self.btn is not None:
            self.btn.config(relief=SUNKEN, disabledforeground="green")
            self.btn['state'] = DISABLED

