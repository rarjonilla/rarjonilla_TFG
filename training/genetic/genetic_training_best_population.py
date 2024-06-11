import random
from typing import List, Dict
import threading


class Genetic_training_best_population:
    """Classe millors individus de la població"""
    def __init__(self, best_n_population: int):
        # Guardar els millors "n" individus
        self.__best_n_population: int = best_n_population
        # Llista amb els millors individus
        self.__best_population: List[Dict] = []
        # Per evitar la concurrencia dels canvis (es dona el cas que varis fils volen afegir un model i intenten eliminar el mateix model de la llista a la vegada, fent que un dels dos falli)
        # Es bloqueja l'accés i hi accediran un a un
        self.__lock = threading.Lock()

    def add_model(self, model_name: str, wins: int, points: int):
        # S'espera a que estigui lliure
        with self.__lock:
            # Dades a emmagatzemar
            new_model: Dict = {"model_name": model_name, "wins": wins, "points": points}

            # Si no hem arribat al limit de poblacio escollida, s'afegeix a la llista
            if len(self.__best_population) < self.__best_n_population:
                self.__best_population.append(new_model)
            else:
                # Si la llista ja té el límit d'elements, eliminem el que tingui menys victories o menys punts en cas que la nova població sigui millor
                worse_population = min(self.__best_population, key=lambda x: (x["wins"], x["points"]))
                if new_model["wins"] > worse_population["wins"] or (new_model["wins"] == worse_population["wins"] and new_model["points"] > worse_population["points"]):
                    # print("s'elimina ", worse_population)
                    # print("s'afegeix ", new_model)
                    self.__best_population.remove(worse_population)
                    self.__best_population.append(new_model)

    def get_best_population(self) -> List[Dict]:
        self.__best_population.sort(key=lambda x: (x['wins'], x['points']), reverse=True)
        return self.__best_population

