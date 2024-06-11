class Suit:
    """Classe Coll"""

    def __init__(self, suit_id: int, label: int) -> None:
        # id del coll (1 a 4)
        self.__id: int = suit_id
        # Text del coll
        self.__label: int = label

    # Getters
    def get_id(self) -> int:
        return self.__id

    def get_label(self) -> int:
        return self.__label

    # Functions
    def is_same_suit(self, suid_id: int) -> bool:
        return self.__id == suid_id

    # Print
    def __str__(self) -> str:
        return str(self.__id) + ' - ' + str(self.__label)


