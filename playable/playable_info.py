from tkinter import *
import tkinter.font as tkFont
from typing import List

from constants import INFO_FRAME_LABEL_SIZE_X, INFO_FRAME_LABEL_SIZE_Y


class Playable_info:
    """
    Classe Playable_info
    """
    def __init__(self, master_frm: Frame) -> None:
        self.master_frm: Frame = master_frm
        self.max_messages: int = 6
        self.messages: List[str] = []

        # test_text = "ashbdshjd jhd gayjwd gyuqwdg yuwqte yqwtge yuqwgr yuqweywt  ruiyq8owru qwo8ru 8qwyr uiqwyr yqwtyuyeuty eywgyeqy r7iwefh jgeaj fgwjqg erwe uiw yehiy ryqe ,nfnasjkhd wkuahedwdwad wqe rqw rqwe qwe qwe wq "

        # self.message = Label(self.master_frm, text="TEST\n\nTEST\n\nTEST\n\nTEST", width=INFO_FRAME_LABEL_SIZE_X, background="white", anchor="w")
        # self.message = Label(self.master_frm, text="TEST\n\n\n\n\n\n", width=INFO_FRAME_LABEL_SIZE_X, height=7, background="white", anchor="w")
        # self.message = Label(self.master_frm, text="TEST\n\nTEST\n\nTEST\n\n", width=INFO_FRAME_LABEL_SIZE_X, height=7, background="white", anchor="w")
        # self.message = Label(self.master_frm, text="TEST\n\nTEST\n\n\n\n", width=INFO_FRAME_LABEL_SIZE_X, height=7, background="white", anchor="w")
        self.message: Label = Label(self.master_frm, text="", width=INFO_FRAME_LABEL_SIZE_X, height=INFO_FRAME_LABEL_SIZE_Y, background="white", anchor=NW, justify="left")
        self.font = tkFont.Font(font=self.message['font']).actual()
        self.message.configure(font=(self.font['family'], 10, 'normal'))
        # self.message = Label(self.master_frm, text=test_text, width=INFO_FRAME_LABEL_SIZE_X, height=INFO_FRAME_LABEL_SIZE_Y, background="red", anchor="w")
        # self.message.pack(padx=10, fill=X, anchor=W)
        self.message.pack(padx=10, fill=X, anchor=W)
        # self.message.grid(column=0, row=0, sticky=W)

    def add_message(self, info: str) -> None:
        self.messages.append(info)

        if len(self.messages) > self.max_messages:
            self.messages.pop(0)

        self.set_text()

    def set_text(self) -> None:
        text = ""
        # for i, t in enumerate(self.messages):
        for i, t in reversed(list(enumerate(self.messages))):
            text += t
            if 1 != 0:
                text += "\n"

        # print("aaa", text, "aaa")

        self.message.config(text=text)

