from typing import Dict, List
from game_environment.game import Non_playable_game


class Reinforcement_training:
    """Classe Supervised Training"""

    def __init__(self, game_type: int, total_episodes: int, num_players: int, single_mode: bool, rules: Dict, models_path: List[str], eps: float, eps_decrease: float, gamma: float, only_one_agent: bool, is_multiple_state: bool = False) -> None:
        # Informació de la partida
        self.__game_type: int = game_type
        self.__total_episodes: int = total_episodes
        self.__num_players: int = num_players
        self.__single_mode: bool = single_mode
        self.__rules = rules

        # Informació dels models
        self.__models_type: List[int] = [9, 9, 9, 9] if not is_multiple_state else [10, 10, 10, 10]
        self.__models_path: List[str] = models_path

        # Informació de l¡entrenament
        self.__only_one_agent: bool = only_one_agent
        self.__is_multiple_state: bool = is_multiple_state

        # Valor d'exploració
        self.__eps: float = eps
        # Decreixement del valor d'exploració
        self.__eps_decrease: float = eps_decrease
        # Valor de descompte
        self.__gamma: float = gamma

        self.__training()

    def __is_brisca(self) -> bool:
        return self.__game_type == 1

    def __training(self):
        # executar la simulació
        game: Non_playable_game = Non_playable_game(self.__game_type, self.__total_episodes, self.__models_type, self.__models_path, self.__num_players, self.__single_mode, self.__rules, True, None, self.__eps, self.__eps_decrease, self.__gamma, self.__only_one_agent)

