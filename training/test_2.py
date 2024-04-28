import concurrent.futures


class Game:
    def __init__(self, datos):
        # Inicializar la partida con los datos proporcionados
        self.datos = datos
        # Simular la partida
        self.simular_partida()

    def simular_partida(self):
        # Aquí iría la lógica de la simulación de la partida
        print("Simulando partida para datos:", self.datos)

    def get_players_wins_points(self):
        # Devolver los resultados de la partida
        # Aquí puedes poner la lógica para calcular los resultados
        return {"player1": 10, "player2": 20, "player3": 15}  # Ejemplo de resultados


# Lista de datos para la simulación (100 en este ejemplo)
datos_simulacion = list(range(100))


# Función para ejecutar la simulación en grupos de 6 datos
def ejecutar_simulacion_concurrentemente():
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = []
        for datos in datos_simulacion:
            # Crear una instancia de la clase Game con los datos
            partida = Game(datos)
            # Tan pronto como una partida haya terminado, ejecutar la función de guardado
            futuro = executor.submit(partida.get_players_wins_points)
            # Agregar la instancia de Game al futuro para poder recuperarla en la devolución de llamada
            futuro.partida = partida
            futuro.add_done_callback(guardar_resultado)
            futures.append(futuro)

            # Si hay 6 futuros activos, esperar a que uno de ellos termine
            if len(futures) == 6:
                concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                futures = []


# Función para guardar los resultados de la simulación
def guardar_resultado(futuro):
    # Recuperar la instancia de Game asociada al futuro
    partida = futuro.partida
    # Obtener los resultados de la partida
    resultados = futuro.result()
    # Aquí puedes agregar la lógica para guardar los resultados en alguna parte
    print("Guardando resultados de la partida para datos:", partida.datos)
    print("Resultados:", resultados)


# Ejecutar la simulación concurrentemente
ejecutar_simulacion_concurrentemente()
