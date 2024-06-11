from typing import Optional


class Genetic_training_cm_log:
    """Classe Log d'encreuament i mutació"""

    def __init__(self, child_name: str, parent_1_name: str, parent_2_name: Optional[str], crossover_ratio: float, crossover_function: Optional[str], mutation_ratio: float, mutation_function: Optional[str]) -> None:
        # Dades a guardar al CSV
        # Nom del model generat
        self.__child_name: str = child_name
        # Nom del model pare 1
        self.__parent_1_name: str = parent_1_name
        # Nom del model pare 2
        self.__parent_2_name: str = parent_2_name
        # Probabilitat d'encreuament
        self.__crossover_ratio: float = crossover_ratio
        # Funció d'encreuament utilitzada
        self.__crossover_function: Optional[str] = crossover_function
        # Probabilitat de mutació
        self.__mutation_ratio: float = mutation_ratio
        # Funció de mutació utilitzada
        self.__mutation_function: Optional[str] = mutation_function

    def get_csv_line(self) -> str:
        return f"'{self.__child_name}', '{self.__parent_1_name}', '{self.__parent_2_name}', '{self.__crossover_ratio}', '{self.__crossover_function}', '{self.__mutation_ratio}', '{self.__mutation_function}'"
