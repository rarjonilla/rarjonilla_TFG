from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.metrics import Precision
from keras.metrics import Precision, Accuracy, Recall


from numpy import array

from configuration import USE_GPU


class Neural_network:
    """Classe xarxa neuronal"""

    def __init__(self, model_type: int, model_path: str) -> None:
        # Informació de la xarxa neuronal
        self.model_type: int = model_type
        self.model_path: str = model_path
        self.model = None

        self.__load_model()

    def __load_model(self) -> None:
        # Es carrega el model
        self.model = load_model(self.model_path, custom_objects={'precision': Precision(), 'recall': Recall(), 'accuracy': Accuracy()})

        # self.__info_layer()

    # Informació de la xarxa neuronal
    def __info_layer(self):
        # resum del model per veure capes i paràmetres
        self.model.summary()

        # Accedir als pesos i biaixos de cada capa
        for capa in self.model.layers:
            print("Capa:", capa.name)
            weights, bias = capa.get_weights()
            print("Weights:", weights)
            print("Bias:", bias)

    def del_model(self) -> None:
        del self.model

    # Model win or lose -> ens prediu 0 o 1
    # Model round points -> ens prediu entre 0 i 5 per la brisca
    # TODO Eliminar aquesta (win or lose)
    def evaluate_model_one_output(self, inputs_array) -> int:
        result = self.model.predict(array([inputs_array]), verbose=0)[0]
        return result

    # Supervisada puntuació i puntuació + heurístic
    def evaluate_model_n_outputs(self, inputs_array) -> Tuple[int, int]:
        # Predicció del model per a un estat del joc concret
        result = self.model.predict(array([inputs_array]), verbose=0)[0]

        # Obtenim la posició del valor màxim
        max_value_position = np.argmax(result)

        return max_value_position, result[max_value_position]

    # Genetic, cada sortida és una possible acció
    def evaluate_model_n_outputs_genetic(self, inputs_array):
        result = self.model.predict(array([inputs_array]), verbose=0)[0]

        return result
