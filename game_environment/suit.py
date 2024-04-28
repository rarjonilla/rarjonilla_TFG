class Suit:
    """
    Classe suit.....
    """

    # def __init__(self, suit: int, id_label: int, value: int, is_king: bool, is_knight: bool) -> None:
    def __init__(self, suit_id: int, label: int) -> None:
        self.__id: int = suit_id
        self.__label: int = label

    # Getters
    def get_id(self) -> int:
        return self.__id

    def get_label(self) -> int:
        return self.__label

    #     def get_one_hot(self) -> str:
    #         one_hot = ""
    #         for i in range(1, 5):
    #             one_hot += "1, " if i == self.__id else "0, "
    #         return one_hot

    # Functions
    def is_same_suit(self, suid_id: int) -> bool:
        return self.__id == suid_id

    # Print
    def __str__(self) -> str:
        return str(self.__id) + ' - ' + str(self.__label)


