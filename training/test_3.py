from typing import List

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model

# Cargar un modelo
modelo_original = load_model("../sl_models/brisca/2j/sl_heu_20240423_205947_1000_partides_layers_50_25.keras")

# modelo_original = Sequential()
# modelo_original.add(Dense(units=10, input_shape=(5,), activation='relu'))  # Capa con 10 neuronas y 5 entradas
# modelo_original.add(Dense(units=8, activation='relu'))  # Capa con 8 neuronas
# modelo_original.add(Dense(units=2, activation='relu'))  # Capa con 8 neuronas
# modelo_original.compile(optimizer='adam', loss='mse')

modelo_original.summary()
num_outputs = modelo_original.layers[-1].units

# Imprimir el número de outputs de la capa final
print("El número de outputs de la capa final es:", num_outputs)

modelo_nuevo = Sequential()
# modelo_nuevo.add(Dense(units=10, input_shape=(5,), activation='relu'))  # Capa con 10 neuronas y 5 entradas
# modelo_nuevo.add(Dense(units=8, activation='relu'))  # Capa con 8 neuronas
# modelo_nuevo.add(Dense(units=2, activation='relu'))  # Capa con 8 neuronas
# modelo_nuevo.compile(optimizer='adam', loss='mse')
modelo_nuevo.add(Dense(units=50, input_shape=(167,), activation='relu'))  # Capa con 10 neuronas y 5 entradas
modelo_nuevo.add(Dense(units=25, activation='relu'))  # Capa con 8 neuronas
modelo_nuevo.add(Dense(84, activation='softmax'))   # Capa de sortida)
modelo_nuevo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# Obtener los pesos y sesgos de todas las capas del modelo original
pesos_totales_m = []
sesgos_totales_m = []

architecture: List[int] = []

for i, capa in enumerate(modelo_original.layers):
    pesos_capa, sesgos_capa = capa.get_weights()

    pesos_totales = []
    sesgos_totales = []

    for j, row in enumerate(pesos_capa):
        # print("Capa", i + 1, j + 1, "Pesos:", row)

        for elemento in row:
            pesos_totales.append(elemento)

        # print("Capa", i + 1, j + 1, "Pesos:", pesos_capa)
        # print(i, j)
    architecture.append(j+1)

    # print("Capa", i + 1, "Sesgos:", sesgos_capa)
    for elemento in sesgos_capa:
        sesgos_totales.append(elemento)

    pesos_totales_m.append(pesos_totales)
    sesgos_totales_m.append(sesgos_totales)

architecture.append(num_outputs)

print(len(pesos_totales_m))
print(len(pesos_totales_m[0]))
print(len(pesos_totales_m[1]))
print(len(pesos_totales_m[2]))

do = True
if do:
    # Convertir lista a np array
    # vector_pesos = np.array(pesos_totales)
    # vector_bias = np.array(sesgos_totales)

    matrius_pesos = []
    matrius_bias = []
    first_index = 0
    last_index = 0

    print("architecture", architecture)
    bias_pos_index = 0
    bias_pos_last = 0
    # La úiltima capa no cal processar-la, ja la tenim en compte en la penúltima iteració
    id_v = 0
    for idx_arc in range(len(architecture) - 1):
        vector_pesos = np.array(pesos_totales_m[id_v])
        vector_bias = np.array(sesgos_totales_m[id_v])

        print("***")
        print(len(vector_pesos))

        matrius_pesos.append(vector_pesos.reshape(architecture[idx_arc], architecture[idx_arc + 1]))
        matrius_bias.append(vector_bias)

        id_v += 1

    print(matrius_pesos)
    print(matrius_bias)

    # Establecer los pesos y sesgos de cada capa
    for capa, peso, sesgo in zip(modelo_nuevo.layers, matrius_pesos, matrius_bias):
        capa.set_weights([peso, sesgo])

    modelo_nuevo.save("sl_heu_test_vectors.keras")