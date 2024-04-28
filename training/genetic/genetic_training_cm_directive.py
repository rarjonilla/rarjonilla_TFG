class Genetic_training_cm_directive:
    def __init__(self, cm_directive: int, custom_crossover_ratio: float, custom_mutation_ratio: float, generations: int):
        # 1- ilm_dhc -> small population
        # 2- dhm_ilc -> large population
        # 3 - fifty-fifty
        # 4 - static ratios -> 0.9 crossover, 0.03 mutation
        # 5 - custom ratios

        self.__cm_directive: int = cm_directive
        # Jo guardo de la generació 1 a la n
        # Si vull aplicar la regla correctament, haig de fer 1 - 1 fins a n - 1 (de la 0 a la n - 1)
        self.__generations: int = generations - 1
        self.__crossover_ratio: float = 0
        self.__mutation_ratio: float = 0

        if cm_directive == 3:
            self.__crossover_ratio = 0.5
            self.__mutation_ratio = 0.5
        elif cm_directive == 4:
            self.__crossover_ratio = 0.9
            self.__mutation_ratio = 0.03
        elif cm_directive == 5:
            self.__crossover_ratio = custom_crossover_ratio
            self.__mutation_ratio = custom_mutation_ratio

    def get_crossover_ratio(self, generation: int) -> float:
        # Jo guardo de la generació 1 a la n
        # Si vull aplicar la regla correctament, haig de fer 1 - 1 fins a n - 1 (de la 0 a la n - 1)
        generation -= 1

        if self.__cm_directive == 1:
            # 1- ilm_dhc -> small population
            # CR = 1 - (LG / Gn)
            # CR -> Crossover Rate, LG = Current generation, Gn = Total number of generations
            self.__crossover_ratio = 1 - (generation / self.__generations)
        elif self.__cm_directive == 2:
            # 2- dhm_ilc -> large population
            # CR = LG / Gn
            self.__crossover_ratio = generation / self.__generations

        return self.__crossover_ratio

    def get_mutation_ratio(self, generation: int) -> float:
        # Jo guardo de la generació 1 a la n
        # Si vull aplicar la regla correctament, haig de fer 1 - 1 fins a n - 1 (de la 0 a la n - 1)
        generation -= 1

        if self.__cm_directive == 1:
            # 1- ilm_dhc -> small population
            # MR = LG / Gn
            # MR -> Mutation Rate, LG = Current generation, Gn = Total number of generations
            self.__mutation_ratio = generation / self.__generations
        elif self.__cm_directive == 2:
            # 2- dhm_ilc -> large population
            # MR = 1 - (LG / Gn)
            self.__mutation_ratio = 1 - (generation / self.__generations)

        return self.__mutation_ratio
