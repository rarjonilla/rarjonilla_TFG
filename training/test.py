import random
from typing import Tuple, List

from training.genetic.genetic_training_crossover import Genetic_training_crossover
from training.genetic.genetic_training_mutation import Genetic_training_mutation

mi_lista = [[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9]]
mi_lista_2 = [[11, 12, 13, 14, 15, 16, 17, 18, 19], [11, 12, 13, 14, 15, 16, 17, 18, 19], [11, 12, 13, 14, 15, 16, 17, 18, 19]]
#mi_lista_2 = [["a", "b", "c", "d", "e", "f", "g", "h", "i"], ["a", "b", "c", "d", "e", "f", "g", "h", "i"], ["a", "b", "c", "d", "e", "f", "g", "h", "i"]]

mi_lista_b = [[11, 12, 13, 14, 15, 16, 17, 18, 19], [11, 12, 13, 14, 15, 16, 17, 18, 19], [11, 12, 13, 14, 15, 16, 17, 18, 19]]
# mi_lista_b_2 = [["j", "k", "l", "m", "n", "o", "p", "q", "r"], ["j", "k", "l", "m", "n", "o", "p", "q", "r"], ["j", "k", "l", "m", "n", "o", "p", "q", "r"]]
mi_lista_b_2 = [[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9]]

gtc = Genetic_training_mutation(1, -1, 1)

c1, method = gtc.mutation((mi_lista, mi_lista_b))


print(method)
print(c1[0])
print(c1[1])
# print(c2[0])
# print(c2[1])
