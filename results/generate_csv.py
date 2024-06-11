import os
from typing import List, Dict

import pandas as pd

from game_environment.game import Non_playable_game


class Generate_csv_results:
    """Classe que s'encarrega de simular partides i emmagatzemar el resultat en CSV"""
    def __init__(self, game_type: int, total_games: int, model_type: List[int], model_path: List[str], num_players: int, single_mode: bool, rules: Dict, csv_file: str):
        # Nom del fitxer on es guardaran les dades
        self.__csv_file: str = csv_file

        # Informació de la simulació
        self.__total_games: int = total_games
        self.__game_type: int = game_type
        self.__model_type: List[int] = model_type
        self.__model_path: List[str] = model_path
        self.__num_players: int = num_players
        self.__single_mode: bool = single_mode
        self.__rules: Dict = rules
        self.__rules_str: str = ''

        # Regles en format String (per afegir-ho al CSV)
        self.__rules_str += '1' if rules['can_change'] else '0'
        self.__rules_str += '1' if rules['last_tens'] else '0'
        self.__rules_str += '1' if rules['black_hand'] else '0'
        if num_players == 2:
            self.__rules_str += '1' if rules['hunt_the_three'] else '0'

        # Inici de la simulacio
        self.__simulate()

    def __simulate(self):
        # executar la simulació
        game: Non_playable_game = Non_playable_game(self.__game_type, self.__total_games, self.__model_type, self.__model_path, self.__num_players, self.__single_mode, self.__rules, False, None)

        # Agafem les victòries i punts
        wins: List[int] = game.score_get_wins()
        points: List[int] = game.score_get_total_scores()

        # S'afegeix el resultat al CSV
        self.__add_result(wins, points)

    def __generate_header(self):
        # Es genera la capçalera segons el nombre de jugadors i segons si és en equip
        if self.__num_players == 2:
            header = ['Player_0', 'Player_1', 'Wins_p0', 'Points_p0', 'Wins_p1', 'Points_p1', 'Rules']
        elif self.__num_players == 3:
            header = ['Player_0', 'Player_1', 'Player_2', 'Wins_p0', 'Points_p0', 'Wins_p1', 'Points_p1', 'Wins_p2', 'Points_p2', 'Rules']
        elif self.__num_players == 4:
            if self.__single_mode:
                header = ['Player_0', 'Player_1', 'Player_2', 'Player_3', 'Wins_p0', 'Points_p0', 'Wins_p1', 'Points_p1', 'Wins_p2', 'Points_p2', 'Wins_p3', 'Points_p3', 'Rules']
            else:
                header = ['Player_0', 'Player_1', 'Player_2', 'Player_3', 'Wins_p0', 'Points_p0', 'Wins_p1', 'Points_p1', 'Rules']

        with open(self.__csv_file, 'w') as f:
            f.write(','.join(header) + '\n')

    def __add_result(self, wins, points):
        # Si el fitxer no existeix, es genera la capçalera
        if not os.path.exists(self.__csv_file):
            self.__generate_header()

        # Es genera la informació de la fila dels resultas
        p1_model_name = self.__model_path[0].split('/')[-1]
        p2_model_name = self.__model_path[1].split('/')[-1]

        if self.__num_players == 2:
            data = {
                'Player_0': [p1_model_name],
                'Player_1': [p2_model_name],
                'Wins_p0': [wins[0]],
                'Points_p0': [points[0]],
                'Wins_p1': [wins[1]],
                'Points_p1': [points[1]],
                'Rules': self.__rules_str
            }
        elif self.__num_players == 3:
            p3_model_name = self.__model_path[2].split('/')[-1]
            data = {
                'Player_0': [p1_model_name],
                'Player_1': [p2_model_name],
                'Player_2': [p3_model_name],
                'Wins_p0': [wins[0]],
                'Points_p0': [points[0]],
                'Wins_p1': [wins[1]],
                'Points_p1': [points[1]],
                'Wins_p2': [wins[2]],
                'Points_p2': [points[2]],
                'Rules': self.__rules_str
            }
        elif self.__num_players == 4 and self.__single_mode:
            p3_model_name = self.__model_path[2].split('/')[-1]
            p4_model_name = self.__model_path[3].split('/')[-1]
            data = {
                'Player_0': [p1_model_name],
                'Player_1': [p2_model_name],
                'Player_2': [p3_model_name],
                'Player_3': [p4_model_name],
                'Wins_p0': wins[0],
                'Points_p0': points[0],
                'Wins_p1': [wins[1]],
                'Points_p1': [points[1]],
                'Wins_p2': [wins[2]],
                'Points_p2': [points[2]],
                'Wins_p3': [wins[3]],
                'Points_p3': [points[3]],
                'Rules': self.__rules_str
            }
        elif self.__num_players == 4 and not self.__single_mode:
            p3_model_name = self.__model_path[2].split('/')[-1]
            p4_model_name = self.__model_path[3].split('/')[-1]

            data = {
                'Player_0': [p1_model_name + '_1'],
                'Player_1': [p2_model_name + '_1'],
                'Player_2': [p3_model_name + '_2'],
                'Player_3': [p4_model_name + '_2'],
                'Wins_p0': wins[0],
                'Points_p0': points[0],
                'Wins_p1': [wins[1]],
                'Points_p1': [points[1]],
                'Rules': self.__rules_str
            }

        df = pd.DataFrame(data)

        # S'afegeix el resultat al CSV
        df.to_csv(self.__csv_file, mode='a', header=False, index=False)