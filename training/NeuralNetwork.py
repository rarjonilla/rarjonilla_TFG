from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.metrics import Precision
from keras.metrics import Precision, Accuracy, Recall


from numpy import array

from configuration import USE_GPU


class Neural_network:

    def __init__(self, model_type: int, model_path: str) -> None:
        self.model_type: int = model_type
        self.model_path: str = model_path
        self.model = None

        self.__load_model()

    def __load_model(self) -> None:
        # self.model = load_model(self.model_path)
        # self.model = load_model(self.model_path, custom_objects={'precision': Precision(), 'recall': Recall(), 'f1_score': tf.keras.metrics.F1Score(), 'accuracy': Accuracy()})
        # print(self.model_path)
        self.model = load_model(self.model_path, custom_objects={'precision': Precision(), 'recall': Recall(), 'accuracy': Accuracy()})

        # self.__info_layer()

    def __info_layer(self):
        # resumen del model per veure capes i paràmetres
        self.model.summary()

        # Acceder a los pesos y sesgos de cada capa
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
        # print(inputs_array)
        result = self.model.predict(array([inputs_array]), verbose=0)[0]
        #  print(result)
        return result

    # Supervisada
    def evaluate_model_n_outputs(self, inputs_array) -> Tuple[int, int]:
        result = self.model.predict(array([inputs_array]), verbose=0)[0]

        # Obtenim la posició del valor màxim
        max_value_position = np.argmax(result)

        # for numero in result:
        #     print("{:.10f}".format(numero))
        # print("")

        return max_value_position, result[max_value_position]
        # print('{:.10f}'.format(result))
        # sum_perdre = result[0] + result[1]
        # return sum_perdre

    def evaluate_model_n_outputs_genetic(self, inputs_array):
        # print(inputs_array)
        result = self.model.predict(array([inputs_array]), verbose=0)[0]

        return result
