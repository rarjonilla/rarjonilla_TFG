from typing import List


class Score:
    """
    Classe suit.....
    """

    def __init__(self, num_players: int, single_mode: bool, max_score: int) -> None:
        self.__single_mode: bool = single_mode

        # Es força una divisió amb resultat enter (4 / 2 = 2.0 i volem que sigui 2)
        self.__num_players: int = num_players if single_mode else num_players // 2

        # Només per playable
        self.__total_players: int = num_players

        self.__scores: List[int] = [0] * num_players

        # Només per playable
        self.__individual_scores: List[int] = [0] * num_players

        self.__history_scores: List[List[int]] = []
        self.__total_scores: List[int] = [0] * num_players
        self.__wins: list[int] = [0] * num_players
        self.__max_score: int = max_score
        self.__last_winners: List[int] = []

    # Getters
    def get_history_scores(self) -> List[List[int]]:
        return self.__history_scores

    def get_individual_scores(self) -> List[int]:
        return self.__individual_scores

    def get_last_winners(self) -> List[int]:
        return self.__last_winners

    def get_total_scores(self) -> List[int]:
        return self.__total_scores

    def get_player_total_scores(self, player_id: int) -> int:
        return self.__total_scores[player_id]

    def get_wins(self) -> List[int]:
        return self.__wins

    def get_player_wins(self, player_id: int) -> int:
        return self.__wins[player_id]

    # Functions
    def add_score(self, player_id: int, score: int) -> None:
        self.__individual_scores[player_id] += score

        if not self.__single_mode:
            player_id %= 2

        self.__scores[player_id] += score

    def finalize_score(self) -> None:
        max_score: int = 0
        for i in range(self.__num_players):
            max_score = max(max_score, self.__scores[i])
            self.__total_scores[i] += self.__scores[i]

        self.__history_scores.append([])

        for i in range(self.__num_players):
            # Un empat a maxim punts és victoria per tots els jugadors que han empatat
            if self.__scores[i] == max_score:
                self.__last_winners.append(i)
                self.__wins[i] += 1

            self.__history_scores[len(self.__history_scores) - 1].append(self.__scores[i])
            self.__scores[i] = 0

        for i in range(self.__total_players):
            self.__individual_scores[i] = 0

    def reset_last_winners(self) -> None:
        self.__last_winners = []

    def set_winners_by_black_hand(self, id_player: int) -> None:
        for i in range(self.__num_players):
            self.__scores[i] = 0

        self.__scores[id_player] = self.__max_score

    def set_winners_by_black_hand_and_tute(self, id_player: int, tutes: List[int]) -> None:
        for i in range(self.__num_players):
            self.__scores[i] = 0

        self.__scores[id_player] = self.__max_score

        for tute in tutes:
            self.__scores[tute] = self.__max_score

    def set_winners_by_hunt_the_three(self, id_player: int) -> None:
        for i in range(self.__num_players):
            self.__scores[i] = 0

        self.__scores[id_player] = self.__max_score

    def set_winners_by_tute(self, tutes: List[int]) -> None:
        if len(tutes) > 0:
            for i in range(self.__num_players):
                self.__scores[i] = 0

            for tute in tutes:
                self.__scores[tute] = self.__max_score

    # Print
    def show_game_score(self) -> None:
        print("show_game_score")
        for i in range(self.__num_players):
            if self.__single_mode:
                print("Player ", i, " ", self.__scores[i], "points")
            else:
                print("Team players ", i, "-", i + 2,  " ", self.__scores[i], "points")
        print("")

    def show_total_game_score(self) -> None:
        print("show_total_game_score")
        for i in range(self.__num_players):
            if self.__single_mode:
                print("Player ", i)
            else:
                print("Team players ", i, "-", i+2)
            print("Total score ", self.__total_scores[i], " points")
            print(self.__wins[i], " wins")
            print("")

    # Print
    def __str__(self) -> str:
        return ""
