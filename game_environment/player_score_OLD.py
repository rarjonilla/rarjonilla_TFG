from typing import List


class Player_score:
    """
    Classe suit.....
    """

    def __init__(self) -> None:
        self.__max_score: int = 130 if self.is_rule_active('last_tens') else 120

        self.__score: int = 0
        self.__history_scores: List[int] = []
        self.__total_score: int = 0
        self.__wins: int = 0

    # Getters
    def get_history_score(self) -> List[int]:
        return self.__history_scores

    def get_total_score(self) -> int:
        return self.__total_score

    def get_wins(self) -> int:
        return self.__wins

    # Functions
    def add_score(self, score: int) -> None:
        self.__score += score

    def finalize_score(self, wins: bool) -> None:
        self.__total_score += self.__score
        self.__history_score.append(self.__score)
        self.__score = 0

        self.__wins += 1 if wins else 0

    def set_winner_by_black_hand_tute_hunt_the_three(self, winner: bool) -> None:
        self.__score = self.__max_score if winner else 0

    # Print
    def __str__(self) -> str:
        return ""
