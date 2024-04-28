from typing import Optional


class Genetic_training_cm_log:
    """Classe Supervised Training"""

    def __init__(self, child_name: str, parent_1_name: str, parent_2_name: Optional[str], crossover_ratio: float, crossover_function: Optional[str], mutation_ratio: float, mutation_function: Optional[str]) -> None:
        self.__child_name: str = child_name
        self.__parent_1_name: str = parent_1_name
        self.__parent_2_name: str = parent_2_name
        self.__crossover_ratio: float = crossover_ratio
        self.__crossover_function: Optional[str] = crossover_function
        self.__mutation_ratio: float = mutation_ratio
        self.__mutation_function: Optional[str] = mutation_function

    def get_csv_line(self) -> str:
        return f"'{self.__child_name}', '{self.__parent_1_name}', '{self.__parent_2_name}', '{self.__crossover_ratio}', '{self.__crossover_function}', '{self.__mutation_ratio}', '{self.__mutation_function}'"
