import gc
import json
import os
import time
from typing import List, Tuple, Optional, Dict

import tensorflow as tf

from constants import DEFAULT_RULES
from configuration import TOTAL_GAMES, NUM_PLAYERS, SINGLE_MODE, GAME_TYPE, MODEL_TYPE, CUSTOM_RULES, \
    MODEL_PATH, USE_GPU, CUSTOM_RULES_2

from game_environment.game import Non_playable_game
from playable.playable import Playable
from training.genetic.genetic_training import Genetic_training
from training.genetic.genetic_training_rl import Genetic_training_rl
from training.reinforcement.reinforcement_training import Reinforcement_training
from training.supervised.supervised_training import Supervised_training

# TODO Avaluacion de modelos. Generar 50 partidas aleatorias n veces y hacer la media de victorias / derrotas
# Generar un grafico de los diferentes modelos
# Por ejemplo, para las supervisadas, los enfrentamientos serian contra la IA de 3 capas automaticas

# TODO MyPy per validació de typing
# TODO Falta fer que la part Playable tingui els elements privats "__variable" i crear les funcions necessàries per al funcionament
# TODO Falta crear una estructura de carpetes millor, on hi hagi els sl_models finals, els sl_models de l'usuari i que estiguin ordenats segons tipus d'entrenament (round points, heuristic, win or lose, genetic (millors poblacions cada X generacions), per reforç, ...)
# TODO -> ga_models -> s'ha de fer una estructura per regles aplicades (per exemple: 0000, 0001, 0010, ... 1111) per saber amb quines regles s'ha entrenat el model
# TODO -> passar is_supervised_training als parametres de "Game_state" i fer proves de que funcioni correctament
# pip install mypy
def simulation(game_type: int = GAME_TYPE, total_games: int = TOTAL_GAMES, model_type: List[int] = MODEL_TYPE, model_path: List[str] = MODEL_PATH, num_players: int = NUM_PLAYERS, single_mode: bool = SINGLE_MODE, is_playable: bool = False, rules: Dict = DEFAULT_RULES, human_player: bool = False) -> None:
    # rules = CUSTOM_RULES if APPLY_CUSTOM_RULES else DEFAULT_RULES

    if not single_mode and num_players < 4:
        raise Exception("only 4 players can play in paired mode")
    if total_games < 1:
        raise Exception("total_games must be, at least, 1")
    if game_type < 1 or game_type > 2:
        raise Exception("There are only 2 game_type available, 1=brisca and 2=tute")

    model_type_error: bool = any(i < 1 or i > 10 for i in model_type)

    if model_type_error:
        raise Exception("There are only 4 model_type available, 1=Random mode, 2=Supervised NN, 3=Genetic NN, 4=Reinforcement NN")
    if num_players < 2 or num_players > 4:
        raise Exception("This game_environment is for 2-4 players only")
    if 'can_change' not in rules:
        raise Exception("The rule 'can_change' has to be set to False or True")
    if 'last_tens' not in rules:
        raise Exception("The rule 'last_tens' has to be set to False or True")
    if 'black_hand' not in rules:
        raise Exception("The rule 'black_hand' has to be set to False or True")
    if 'hunt_the_three' not in rules:
        raise Exception("The rule 'hunt_the_three' has to be set to False or True")
    if 'only_assist' not in rules:
        raise Exception("The rule 'only_assist' has to be set to False or True")

    if is_playable:
        # Playable_game(game_id, game_type, total_games, model_type, num_players, single_mode, rules)
        # TODO Fer que human_player sigui la variable que es demana en el main (pasar-la aqui per parametre i pasar-la al Playable)
        # Fer els canvis necessaris
        Playable(game_type, human_player, total_games, model_type, model_path, num_players, single_mode, rules, False, None)
    else:
        Non_playable_game(game_type, total_games, model_type, model_path, num_players, single_mode, rules, False, None)


def training():
    training_type: int = 3
    # total_games = 500
    total_games = 8000
    # total_games = 1500
    # total_games = 1
    train = False

    # model_ = f'sl_models/brisca/{i}j/st_nds_heu_20240403_202253.h5'
    for i in range(2, 5):
        # if i == 2:
        # tg = total_games if i == 2 else total_games * 2
        tg = total_games

        save_filename: str = f'{tg}_partides_40_20'
        do_this = True
        if i > 2:
            print(f"brisca single {i} players")
            start_time_brisca = time.time()
            Supervised_training(training_type, total_games=tg, game_type=1, num_players=i, single_mode=True, only_assist=False, rivals_model_type=[1, 1, 1, 1], rivals_model_name=[None, None, None, None], generate_data=True, csv_filename=None, csv_filename_2=None, save_prepared_data=True, save_filename=save_filename, do_training=train, layers=[40, 20])
            print("--- %s seconds ---" % (time.time() - start_time_brisca))

            print(f"tute single no assist {i} players")
            start_time_tute = time.time()
            Supervised_training(training_type, total_games=tg, game_type=2, num_players=i, single_mode=True, only_assist=False, rivals_model_type=[1, 1, 1, 1], rivals_model_name=[None, None, None, None], generate_data=True, csv_filename=None, csv_filename_2=None, save_prepared_data=True, save_filename=save_filename, do_training=train, layers=[40, 20])
            print("--- %s seconds ---" % (time.time() - start_time_tute))

        print(f"tute single assist {i} players")
        start_time_tute = time.time()
        Supervised_training(training_type, total_games=tg, game_type=2, num_players=i, single_mode=True, only_assist=True, rivals_model_type=[1, 1, 1, 1], rivals_model_name=[None, None, None, None], generate_data=True, csv_filename=None, csv_filename_2=None, save_prepared_data=True, save_filename=save_filename, do_training=train, layers=[40, 20])
        print("--- %s seconds ---" % (time.time() - start_time_tute))

        if i == 4:
            if do_this:
                print(f"brisca team {i} players")
                start_time_brisca = time.time()
                # Supervised_training(training_type, total_games=tg, game_type=1, num_players=i, single_mode=False, only_assist=False, rivals_model_type=[1, 1, 1, 1], rivals_model_name=[None, None, None, None], generate_data=True, csv_filename=None, prepare_data=True, train=True)
                # Supervised_training(training_type, total_games=tg, game_type=1, num_players=i, single_mode=False, only_assist=False, rivals_model_type=[1, 1, 1, 1], rivals_model_name=[None, None, None, None], generate_data=False, csv_filename='data/brisca/2j/20240401_215008.csv', prepare_data=True, train=True)
                Supervised_training(training_type, total_games=tg, game_type=1, num_players=i, single_mode=False, only_assist=False, rivals_model_type=[1, 1, 1, 1], rivals_model_name=[None, None, None, None], generate_data=True, csv_filename=None, csv_filename_2=None, save_prepared_data=True, save_filename=save_filename, do_training=train, layers=[40, 20])
                print("--- %s seconds ---" % (time.time() - start_time_brisca))

            print(f"tute team no assist {i} players")
            start_time_tute = time.time()
            Supervised_training(training_type, total_games=tg, game_type=2, num_players=i, single_mode=False, only_assist=False, rivals_model_type=[1, 1, 1, 1], rivals_model_name=[None, None, None, None], generate_data=True, csv_filename=None, csv_filename_2=None, save_prepared_data=True, save_filename=save_filename, do_training=train, layers=[40, 20])
            print("--- %s seconds ---" % (time.time() - start_time_tute))

            print(f"tute team assist {i} players")
            start_time_tute = time.time()
            Supervised_training(training_type, total_games=tg, game_type=2, num_players=i, single_mode=False, only_assist=True, rivals_model_type=[1, 1, 1, 1], rivals_model_name=[None, None, None, None], generate_data=True, csv_filename=None, csv_filename_2=None, save_prepared_data=True, save_filename=save_filename, do_training=train, layers=[40, 20])
            print("--- %s seconds ---" % (time.time() - start_time_tute))


def ask_yes_no(question: str) -> bool:
    response = False

    while True:
        input_yes_or_no = input(question)
        if input_yes_or_no == "" or input_yes_or_no.lower() == "s":
            response = True
            break
        elif input_yes_or_no.lower() == "n":
            break

    return response


def ask_int_options(question: str, min_value: int, max_value: int):
    while True:
        try:
            number: int = int(input(question))
            if min_value <= number <= max_value:
                return number
            else:
                print(f"Si us plau, introdueix un número entre {min_value} i {max_value}.")
        except ValueError:
            print("Si us plau, introdueix un número enter vàlid.")


def ask_float_options(question: str, min_value: int, max_value: int):
    while True:
        try:
            number: float = float(input(question))
            if min_value <= number <= max_value:
                return number
            else:
                print(f"Si us plau, introdueix un número entre {min_value} i {max_value}.")
        except ValueError:
            print("Si us plau, introdueix un número decimal vàlid.")


def ask_int_options_multiple(question: str, initial_value: int, less_than_or_equal: Optional[int] = None):
    while True:
        try:
            number = int(input(question))
            if number % initial_value == 0:
                if less_than_or_equal is not None and number > less_than_or_equal:
                    print(f"Si us plau, introdueix un número múltiple de {initial_value} i inferior a {less_than_or_equal}.")
                else:
                    return number
            else:
                print(f"Si us plau, introdueix un número múltiple de {initial_value}.")
        except ValueError:
            print("Si us plau, introdueix un número enter vàlid.")


def ask_int_options_old(question: str, answers: List[int]) -> int:
    while True:
        try:
            response: int = int(input(question))
            if response in answers:
                break
            else:
                print("El valor introduït no és vàlid")
        except ValueError:
            print("El valor introduït no és vàlid")

    return response


def ask_int(question: str) -> int:
    while True:
        try:
            response: int = int(input(question))
            break
        except ValueError:
            print("El valor introduït no és vàlid")

    return response


def file_selection(directory: str, model_prefix: Optional[str] = None):
    # Obtenir la llista d'arxius del directorio
    all_files = os.listdir(directory)

    # Filtrar els arxius que comencen amb el prefix del tipus d'entrenament
    files = [file for file in all_files if file.startswith(model_prefix)] if model_prefix is not None else all_files

    if len(files) > 0:
        # Mostrar els arxius disponibles
        print("Arxius disponibles:")
        for i, file in enumerate(files):
            print(f"{i + 1}. {file}")

        print(f"{i + 2}. Tornar endarrere")

        # Selecció de model de l'usuari
        try:
            model_selected = int(input("Escull l'arxiu a utilitzar: ")) - 1

            # Validar la selección del usuario
            if 0 <= model_selected < len(files):
                return files[model_selected]
            elif model_selected == len(files):
                return None
            else:
                print("Selecció no vàlida. Si us plau, esculli una opció vàlida")
                return file_selection(directory, model_prefix)
        except ValueError:
            print("Selecció no vàlida. Si us plau, esculli una opció vàlida")
            return None
    else:
        print("El directori no conté cap arxiu, esculli una altra opció")
        return None


def ask_models(directory: str, human_player: bool, num_players: int, training_type: Optional[int] = None) -> Tuple[List[int], List[str]]:
    model_type: List[int] = []
    model_path: List[str] = []
    i: int = 0
    while True:
        if i == num_players:
            break
    # for i in range(num_players):

        if human_player and i == 0:
            model_type.append(1)
            model_path.append(None)
            i += 1
        else:
            if training_type is None:
                mt: int = ask_int_options(f"Indiqui el tipus de model que vol carregar per al jugador {i + 1} (1- Aleatòri -- 2- Aprenentatge supervisat -- 3- Algoritme genètic 4- Aprenentatge per reforç: ", 1, 4)
            else:
                mt = 3

            if mt == 1:
                model_type.append(1)
                model_path.append(None)
                i += 1
            elif mt == 2:
                model_dir: str = directory.replace("xxxx_models", "sl_models")
                # prefix = 'sl'
                #                 if "_wl_" in model:
                #                     model_type.append(5)
                #                 elif "_rp_" in model:
                #                     model_type.append(4)
                #                 elif "_heu_" in model:
                #                     model_type.append(6)
            elif mt == 3:
                model_dir = directory.replace("xxxx_models", "ga_models")
                # prefix = 'ga'
                # model_type.append(9999999)
            elif mt == 4:
                model_dir = directory.replace("xxxx_models", "rl_models")
                # prefix = 'rl'
                # model_type.append(9999999)

            if mt != 1:
                model = file_selection(model_dir)
                if model is not None:
                    model = model_dir + "/" + model
                    if mt == 2:
                        if "_wl_" in model:
                            model_type.append(5)
                        elif "_rp_" in model:
                            model_type.append(4)
                        elif "_heu_" in model:
                            if "_norm_" in model:
                                model_type.append(8)
                            else:
                                model_type.append(6)

                        model_path.append(model)
                    elif mt == 3:
                        while True:
                            final_model = file_selection(model + "/best_models/")
                            print(final_model)
                            print(model + "/best_models/" + final_model)
                            if final_model is not None:
                                model_type.append(7)
                                model_path.append(model + "/best_models/" + final_model)
                                break
                    elif mt == 4:
                        if "_mc_multiple_" in model:
                            model_type.append(10)
                        else:
                            model_type.append(9)
                        model_path.append(model)

                    i += 1

    # print(model_type, model_path)
    return model_type, model_path


def ask_game_type() -> int:
    return ask_int_options("Indiqui el tipus de joc (1- Brisca -- 2- Tute: ", 1, 2)


def ask_num_players() -> int:
    return ask_int_options("Indiqui el nombre de jugadors (2 -- 3 -- 4): ", 2, 4)


def ask_total_games() -> int:
    return ask_int("Indiqui el total de jocs a simular: ")


def ask_single_mode() -> bool:
    return ask_yes_no("Vol simular partides en modalitat individual? (S/n): ")


# def ask_simulation_variables(is_playable: bool, ask_for_models: bool) -> Tuple[int, int, int, List[int], List[str], bool]:
#     game_type: int = ask_int_options("Indiqui el tipus de joc - Brisca (1) / Tute (2): ", [1, 2])
#     num_players: int = ask_int_options("Indiqui el nombre de jugadors (2/3/4): ", [2, 3, 4])
#     single_mode: bool = SINGLE_MODE
#
#     if num_players == 4:
#         single_mode = ask_yes_no("Vol simular partides en modalitat individual? (S/n): ")
#
#     total_games: int = TOTAL_GAMES
#     if not is_playable:
#         total_games = ask_int("Indiqui el total de jocs a simular: ")
#
#     model_type: List[int] = None
#     model_path: List[Optional[str]] = None
#
#     if ask_for_models:
#         directory: str = "sl_models/"
#         if game_type == 1:
#             directory += "brisca/"
#         else:
#             directory += "tute/"
#
#         directory += f"{num_players}j"
#
#         if not single_mode:
#             directory += "t"
#
#         model_type, model_path = ask_models(directory, is_playable, num_players)
#
#     return game_type, num_players, total_games, model_type, model_path, single_mode


def ask_rules(game_type: int, num_players: int, all_rules: bool) -> Tuple[Dict]:
    rule_can_change: bool = False
    rule_last_tens: bool = False
    rule_black_hand: bool = False
    rule_hunt_the_three: bool = False
    rule_only_assist: bool = False

    if all_rules:
        rule_can_change = ask_yes_no("Vol aplicar la regla d'intercanviar carta de triomf? (S/n): ")
        rule_last_tens = ask_yes_no("Vol aplicar la regla de les 10 d'últimes? (S/n): ")
        rule_black_hand = ask_yes_no("Vol aplicar la regla de la mà negra? (S/n): ")

        if num_players == 2:
            rule_hunt_the_three = ask_yes_no("Vol aplicar la regla de caça del 3? (S/n): ")

    if game_type == 2:
        rule_only_assist = ask_yes_no("Vol aplicar la regla de només assistir per obligació? (S/n): ")

    rules: Dict = {
        'can_change': rule_can_change,
        'last_tens': rule_last_tens,
        'black_hand': rule_black_hand,
        'hunt_the_three': rule_hunt_the_three,
        'only_assist': rule_only_assist,
    }

    return rules


def main_non_playable() -> None:
    game_type: int = GAME_TYPE
    num_players: int = NUM_PLAYERS
    total_games: int = TOTAL_GAMES
    model_type: List[int] = MODEL_TYPE
    model_path: List[str] = MODEL_PATH
    single_mode: bool = SINGLE_MODE
    is_playable: bool = False
    human_player: bool = False
    rules: Dict = CUSTOM_RULES

    is_default = ask_yes_no("Vol utilitzar la configuració per defecte? (S/n): ")

    if not is_default:
        game_type = ask_game_type()
        num_players = ask_num_players()

        if num_players == 4:
            single_mode = ask_single_mode()

        total_games = ask_total_games()

        rules = ask_rules(game_type, num_players, True)

        directory: str = "xxxx_models/"
        if game_type == 1:
            directory += "brisca/"
        elif rules['only_assist']:
            directory += "tute_only_assist/"
        else:
            directory += "tute/"

        directory += f"{num_players}j"

        if not single_mode:
            directory += "t"

        model_type, model_path = ask_models(directory, human_player, num_players)

    simulation(game_type=game_type, num_players=num_players, is_playable=is_playable, total_games=total_games,
               single_mode=single_mode, model_type=model_type, model_path=model_path, rules=rules,
               human_player=human_player)


def main_playable() -> None:
    game_type: int = GAME_TYPE
    num_players: int = NUM_PLAYERS
    total_games: int = TOTAL_GAMES
    model_type: List[int] = MODEL_TYPE
    model_path: List[str] = MODEL_PATH
    single_mode: bool = SINGLE_MODE
    is_playable: bool = True
    human_player: bool = False
    rules: Dict = CUSTOM_RULES

    is_default = ask_yes_no("Vol utilitzar la configuració per defecte? (S/n): ")

    if not is_default:
        game_type = ask_game_type()
        num_players = ask_num_players()

        if num_players == 4:
            single_mode = ask_single_mode()

        human_player = ask_yes_no("Vol que un jugador humà participi? (S/n): ")

        rules = ask_rules(game_type, num_players, True)

        directory: str = "xxxx_models/"
        if game_type == 1:
            directory += "brisca/"
        elif rules['only_assist']:
            directory += "tute_only_assist/"
        else:
            directory += "tute/"

        directory += f"{num_players}j"

        if not single_mode:
            directory += "t"

        model_type, model_path = ask_models(directory, human_player, num_players)

    simulation(game_type=game_type, num_players=num_players, is_playable=is_playable, total_games=total_games,
               single_mode=single_mode, model_type=model_type, model_path=model_path, rules=rules,
               human_player=human_player)


def main_training_sl(training_type: int) -> None:
    game_type: int = GAME_TYPE
    num_players: int = NUM_PLAYERS
    total_games: int = TOTAL_GAMES
    model_type: List[int] = MODEL_TYPE
    model_path: List[str] = MODEL_PATH
    single_mode: bool = SINGLE_MODE
    rules: Dict = CUSTOM_RULES

    generate_data: bool = True
    csv_filename: str = None
    save_filename: str = None
    save_prepared_data: bool = False

    game_type = ask_game_type()
    num_players = ask_num_players()

    if num_players == 4:
        single_mode = ask_single_mode()

    rules = ask_rules(game_type, num_players, False)

    while True:
        generate_data = ask_yes_no("Vol generar les dades de zero? (S/n): ")

        if not generate_data:
            directory: str = "data/"
            if game_type == 1:
                directory += "brisca/"
            else:
                directory += "tute/"

            directory += f"{num_players}j"

            if not single_mode:
                directory += "t"

            filename: str = file_selection(directory, None)
            if filename is not None:
                csv_filename = directory + "/" + filename

                if 'prepared' not in csv_filename:
                    save_prepared_data = ask_yes_no("Vol emmagatzemar el conjunt de dades sense comprimir? (S/n): ")

                break
        else:
            total_games = ask_total_games()

            directory: str = "sl_models/"
            if game_type == 1:
                directory += "brisca/"
            elif rules['only_assist']:
                directory += "tute_only_assist/"
            else:
                directory += "tute/"

            directory += f"{num_players}j"

            if not single_mode:
                directory += "t"

            model_type, model_path = ask_models(directory, False, num_players)

            save_prepared_data = ask_yes_no("Vol emmagatzemar el conjunt de dades sense comprimir? (S/n): ")
            break

    save_filename = input("Indiqui el nom amb el qual vol emmagatzemar el model: ")

    # Total de capes
    num_layers = ask_int_options("Introdueix el nombre de capes intermèdies (mínim 1): ", 1, float('inf'))

    # Neurones per capa
    layers: List[int] = []

    for i in range(num_layers):
        num_layer = ask_int_options(f"Introdueix el nombre de neurones per la capa {i + 1} (mínim 1): ", 1,
                                    float('inf'))
        layers.append(num_layer)

    # training(training_type)
    Supervised_training(training_type, total_games=total_games, game_type=game_type,
                        num_players=num_players, single_mode=single_mode,
                        only_assist=rules['only_assist'], rivals_model_type=model_type,
                        rivals_model_name=model_path, generate_data=generate_data,
                        csv_filename=csv_filename, csv_filename_2=None,
                        save_prepared_data=save_prepared_data, save_filename=save_filename,
                        layers=layers)
        # Supervised_training(training_type, total_games=total_games, game_type=game_type, num_players=num_players, single_mode=single_mode, only_assist=rules['only_assist'], rivals_model_type=model_type, rivals_model_name=model_path, generate_data=generate_data, csv_filename=csv_filename, csv_filename_2=None, prepare_data=False, train=True)


def main_training_ga() -> None:
    game_type: int = GAME_TYPE
    num_players: int = NUM_PLAYERS
    total_games: int = TOTAL_GAMES
    model_type: List[int] = MODEL_TYPE
    model_path: List[str] = MODEL_PATH
    single_mode: bool = SINGLE_MODE
    rules: Dict = CUSTOM_RULES

    game_type = ask_game_type()
    num_players = ask_num_players()

    if num_players == 4:
        single_mode = ask_single_mode()

    rules = ask_rules(game_type, num_players, False)

    # Si es tracta de Genetic Algorithm, cal demanar que vol fer (començar de 0, seguir entrenament, seguir amb entrenament finalitzat)
    ga_training_start_from_zero: bool = ask_yes_no("Vols començar un nou entrenament? (S/n): ")

    ask_training_info: bool = True
    data: Dict = {}

    if not ga_training_start_from_zero:
        while True:
            # Demanar quin model vol seguir entrenant
            directory: str = "ga_models/"
            if game_type == 1:
                directory += "brisca/"
            elif rules['only_assist']:
                directory += "tute_only_assist/"
            else:
                directory += "tute/"

            directory += f"{num_players}j"

            filename: str = file_selection(directory, None)
            if filename is not None:
                break

        # Comprovar si ja està acabat l'entrenament, si és així, cal demanar les dades pel nou entrenament
        with open(directory + "/" + filename + '/training_info.json', 'r') as f:
            data = json.load(f)
            if data["generations"] != data["generation"]:
                ask_training_info = False
    else:
        # Total de capes
        num_layers = ask_int_options("Introdueix el nombre de capes intermèdies (mínim 1): ", 1, float('inf'))

        # Neurones per capa
        layers: List[int] = []

        for i in range(num_layers):
            num_layer = ask_int_options(f"Introdueix el nombre de neurones per la capa {i + 1} (mínim 1): ", 1, float('inf'))
            layers.append(num_layer)

    print(ga_training_start_from_zero, ask_training_info)

    if ga_training_start_from_zero or ask_training_info:
        # TODO demanar sl_model

        save_filename: str = input("Indiqui el nom amb el qual vol emmagatzemar el model: ")
        total_games = ask_total_games()
        population_size: int = ask_int_options_multiple(f"Introdueix el nombre de poblacions (múltiple de {num_players}): ", num_players)

        if ga_training_start_from_zero:
            threads_q: bool = ask_yes_no("Vols utilitzar el màxim de fils possibles? (S/n): ")
            # if sl_demanat:
            #   threads: int = population_size
            # else:
            threads: int = population_size / num_players

            if not threads_q:
                threads = ask_int_options_multiple(f"Introdueix el nombre de fils (múltiple de {num_players} i inferior o igual a {threads}): ", num_players, threads)

            double_tournament: bool = ask_yes_no("Vols realitzar una doble selecció de les millors poblacions (Es farà una primera selecció de les n*2 millors poblacions i es tornaran a fer emparellaments per seleccionar les n millors.)? (S/n): ")

        best_n_population = ask_int_options(f"Introdueix el nombre millors poblacions a seleccionar de cada generació (mínim 1): ", 1, population_size)
        generations: int = ask_int_options(f"Introdueix el nombre de generacions totals (mínim 1): ", 1, float('inf'))
        cm_directive: int = ask_int_options(f"Introdueix la directiva de ratios d'encreuament/mutació a seguir (1- DHM-ILC -- 2- ILM-DHC -- 3- Fifty-Fifty -- 4- 0.9 encreuament / 0.03 mutació -- 5- Custom ratios ): ", 1, 5)
        custom_crossover_ratio: float = 0
        custom_mutation_ratio: float = 0
        elite_selection: bool = ask_yes_no("Vols aplicar la tècnica de la selecció d'èlit? (S/n): ")

        if cm_directive == 5:
            custom_crossover_ratio = ask_float_options(f"Introdueix el ratio d'encreuament (entre 0 i 1): ", 0, 1)
            custom_mutation_ratio = ask_float_options(f"Introdueix el ratio de mutació (entre 0 i 1): ", 0, 1)

    if ga_training_start_from_zero:
        # Començar entrenament
        # TODO demanar sl_model
        Genetic_training(game_type, total_games, num_players, single_mode, rules, False, False, save_filename, layers, population_size, best_n_population, generations, 0, cm_directive, custom_crossover_ratio, custom_mutation_ratio, True, threads, double_tournament, 'sl_models/brisca/2j/sl_heu_20240423_205947_1000_partides_layers_50_25.keras', elite_selection)
    else:
        if ask_training_info:
            # TODO demanar sl_model
            # TODO -> hem de pasar el nom nou del model i l'antic (es pot donar el cas que volguem canviar-li el nom, més poblacions, ...)
            # Començar entrenament partint dels models ja entrenats d'un altre entrenament
            Genetic_training(data["game_type"], data["total_games"], data["num_players"], data["single_mode"], data["rules"], False, True, data["save_filename"], data["layers"], population_size, best_n_population, generations, 0, cm_directive, custom_crossover_ratio, custom_mutation_ratio, True, data["threads"], data["double_tournament"], data["sl_model"], None, elite_selection)
        else:
            # Seguir entrenament
            Genetic_training(data["game_type"], data["total_games"], data["num_players"], data["single_mode"], data["rules"], True, False, data["save_filename"], data["layers"], data["population_size"], data["best_n_population"], data["generations"], data["generation"], data["cm_directive"], data["custom_crossover_ratio"], data["custom_mutation_ratio"], data["finalized_generation"], data["threads"], data["double_tournament"], data["sl_model"], data["elite_selection"])


def main() -> None:
    while True:
        print("1. Simulació de partides")
        print("2. Simulació de partides GUI")
        print("3. Entrenament d'un agent")
        print("4. Entrenament d'agents de totes les modalitats (test, no hi serà a l'entrega final)")
        print("5. Sortir")
        option: int = ask_int_options("Indiqui l'acció a realitzar: ", 1, 5)

        if option == 1:
            main_non_playable()
        if option == 2:
            main_playable()
        if option == 3:
            print("1. Supervised Training - Round points")
            print("2. Supervised Training - Win or Lose")
            print("3. Supervised Training - Round points + heuristic")
            print("4. All Supervised Training from same generated data")
            print("5. Genetic Algorithm")
            print("6. Reinforcement Learning")
            training_type = ask_int_options("Indiqui el tipus de model a generar: ", 1, 5)
            if training_type < 5:
                main_training_sl(training_type)
            if training_type == 5:
                main_training_ga()
        elif option == 5:
            print("Sortint del programa...")
            break
        else:
            print("Opció no válida. Si us plau, introdueixi una opció vàlida.")


start_time = time.time()
a = True
if USE_GPU and a:
    # Configurar Tensorflow para que utilice la GPU si está disponible
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Configurar para que Tensorflow utilice toda la memoria de la GPU de manera dinámica
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # Establecer la preferencia para ejecutar en la GPU en lugar de en la CPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            print("Configuració de GPU completada.")
        except RuntimeError as e:
            print(e)

# main()
# training()

# Genetic_training(1, 10, 2, True, CUSTOM_RULES, 'test_1', [75, 50], 50, 10, 3, 1, None, None)
# with open('ga_models/brisca/2j/0000_20240515_162533_10_games_1000_generations_layers_50_25_population_32_best_6_directive_1_elite_selection/training_info.json', 'r') as f:
    # data: Dict = json.load(f)
    # Seguir entrenament
    # Genetic_training(data["game_type"], data["total_games"], data["num_players"], data["single_mode"], data["rules"], True, False, data["save_filename"], data["layers"], data["population_size"], data["best_n_population"], data["generations"], data["generation"], data["cm_directive"], data["custom_crossover_ratio"], data["custom_mutation_ratio"], data["finalized_generation"], data["threads"], data["double_tournament"], data["sl_model"], data["elite_selection"])

    # Començar entrenament partint dels sl_models ja entrenats d'un altre entrenament
    # Genetic_training(data["game_type"], data["total_games"], data["num_players"], data["single_mode"], data["rules"], False, True, data["save_filename"], data["layers"], data["population_size"], 2, 10, 0, 1, None, None, True, data["threads"], data["double_tournament"], data["sl_model"], True)

# Començar entrenament
# Genetic_training(1, 10, 2, True, CUSTOM_RULES, False, False, '10_games_1000_generations_layers_50_25_population_32_best_6_directive_1_elite_selection', [50, 25], 32, 6, 1000, 0, 1, None, None, True, 32, False, None, True)

# RL
# Reinforcement_training(1, 500000, 2,  True, CUSTOM_RULES, ['rl_models/brisca/2j/0000_20240501_222500_500000_partides_eps_005_gamma_1_negative_points_agent_1_continue_to_1000000', 'rl_models/brisca/2j/0000_20240501_222500_500000_partides_eps_005_gamma_1_negative_points_agent_2_continue_to_1000000', None, None])

# Macro entrenament (només un agent que juga contra si mateix i aprèn dels dos jugadors alhora)
# episodes = 3000000
# a = Reinforcement_training(1, episodes, 2, True, CUSTOM_RULES, [f'rl_models/brisca/2j/0000_20240508_073000_5000000_partides_eps_01_gamma_1_negative_points_only_one_agent_limited_inputs', None, None, None], 0.1, 1e-7, 1.0, True, False)
# a = Reinforcement_training(1, episodes, 2, True, CUSTOM_RULES, [f'rl_models/brisca/2j/0000_20240508_073000_5000000_partides_eps_01_gamma_1_negative_points_only_one_agent_limited_inputs', None, None, None], 0.1, 1e-7, 1.0, True, True)
# a = None
# gc.collect()
# episodes = 3000000
# a = Reinforcement_training(1, episodes, 2, True, CUSTOM_RULES, [f'rl_models/brisca/2j/0000_20240511_164800_2000000_partides_eps_01_gamma_1_negative_points_only_one_agent_limited_inputs_no_decrease_eps_simple_state', None, None, None], 0.1, 0, 1.0, True, False)
# episodes = 100
# a = Reinforcement_training(1, episodes, 2, True, CUSTOM_RULES, [f'rl_models/brisca/2j/0000_20240512_090600_2000000_partides_eps_01_gamma_1_negative_points_only_one_agent_limited_inputs_no_decrease_eps_mc_multiple_', None, None, None], 0.1, 0, 1.0, True, True)

# Proves -> no funciona bé
do_this = False
if do_this:
    for i in range(2, 5):
        episodes: int = 500000
        a = Reinforcement_training(1, episodes, i, True, CUSTOM_RULES, [
            f'rl_models/brisca/{i}j/0000_20240503_073700_{episodes}_partides_eps_005_gamma_1_negative_points_agent_1',
            f'rl_models/brisca/{i}j/0000_20240503_073700_{episodes}_partides_eps_005_gamma_1_negative_points_agent_2',
            f'rl_models/brisca/{i}j/0000_20240503_073700_{episodes}_partides_eps_005_gamma_1_negative_points_agent_3',
            f'rl_models/brisca/{i}j/0000_20240503_073700_{episodes}_partides_eps_005_gamma_1_negative_points_agent_4'], 0.05, 1e-7, 1.0, False, False)
        a = Reinforcement_training(2, episodes, i, True, CUSTOM_RULES, [
            f'rl_models/tute/{i}j/0000_20240503_073700_{episodes}_partides_eps_005_gamma_1_negative_points_agent_1',
            f'rl_models/tute/{i}j/0000_20240503_073700_{episodes}_partides_eps_005_gamma_1_negative_points_agent_2',
            f'rl_models/tute/{i}j/0000_20240503_073700_{episodes}_partides_eps_005_gamma_1_negative_points_agent_3',
            f'rl_models/tute/{i}j/0000_20240503_073700_{episodes}_partides_eps_005_gamma_1_negative_points_agent_4'], 0.05, 1e-7, 1.0, False, False)

        a = Reinforcement_training(2, episodes, i, True, CUSTOM_RULES_2, [
            f'rl_models/tute_only_assist/{i}j/0000_20240503_073700_{episodes}_partides_eps_005_gamma_1_negative_points_agent_1',
            f'rl_models/tute_only_assist/{i}j/0000_20240503_073700_{episodes}_partides_eps_005_gamma_1_negative_points_agent_2',
            f'rl_models/tute_only_assist/{i}j/0000_20240503_073700_{episodes}_partides_eps_005_gamma_1_negative_points_agent_3',
            f'rl_models/tute_only_assist/{i}j/0000_20240503_073700_{episodes}_partides_eps_005_gamma_1_negative_points_agent_4'], 0.05, 1e-7, 1.0, False, False)

        if i == 4:
            a = Reinforcement_training(1, episodes, i, False, CUSTOM_RULES, [
                f'rl_models/brisca/{i}jt/0000_20240503_073700_{episodes}_partides_eps_005_gamma_1_negative_points_agent_1',
                f'rl_models/brisca/{i}jt/0000_20240503_073700_{episodes}_partides_eps_005_gamma_1_negative_points_agent_2',
                f'rl_models/brisca/{i}jt/0000_20240503_073700_{episodes}_partides_eps_005_gamma_1_negative_points_agent_3',
                f'rl_models/brisca/{i}jt/0000_20240503_073700_{episodes}_partides_eps_005_gamma_1_negative_points_agent_4'], 0.05, 1e-7, 1.0, False, False)
            a = Reinforcement_training(2, episodes, i, False, CUSTOM_RULES, [
                f'rl_models/tute/{i}jt/0000_20240503_073700_{episodes}_partides_eps_005_gamma_1_negative_points_agent_1',
                f'rl_models/tute/{i}jt/0000_20240503_073700_{episodes}_partides_eps_005_gamma_1_negative_points_agent_2',
                f'rl_models/tute/{i}jt/0000_20240503_073700_{episodes}_partides_eps_005_gamma_1_negative_points_agent_3',
                f'rl_models/tute/{i}jt/0000_20240503_073700_{episodes}_partides_eps_005_gamma_1_negative_points_agent_4'], 0.05, 1e-7, 1.0, False, False)

            a = Reinforcement_training(2, episodes, i, False, CUSTOM_RULES_2, [
                f'rl_models/tute_only_assist/{i}jt/0000_20240503_073700_{episodes}_partides_eps_005_gamma_1_negative_points_agent_1',
                f'rl_models/tute_only_assist/{i}jt/0000_20240503_073700_{episodes}_partides_eps_005_gamma_1_negative_points_agent_2',
                f'rl_models/tute_only_assist/{i}jt/0000_20240503_073700_{episodes}_partides_eps_005_gamma_1_negative_points_agent_3',
                f'rl_models/tute_only_assist/{i}jt/0000_20240503_073700_{episodes}_partides_eps_005_gamma_1_negative_points_agent_4'], 0.05, 1e-7, 1.0, False, False)

# a = Genetic_training_rl(1, 10, 2, True, CUSTOM_RULES, False, False, '10_games_200_population_best_20_10000_generation_005_eps_rl', 200, 20, 10000, 0, 1, None, None, True, 500, False, None, True, 0.05, 1e-7, 1.0, False)

# with open('ga_models/brisca/2j/0000_20240513_072948_10_games_200_population_best_20_10000_generation_005_eps_rl/training_info.json', 'r') as f:
    # data: Dict = json.load(f)
    # Seguir entrenament
    # Genetic_training_rl(data["game_type"], data["total_games"], data["num_players"], data["single_mode"], data["rules"], True, False, data["save_filename"], data["population_size"], data["best_n_population"], data["generations"], data["generation"], data["cm_directive"], data["custom_crossover_ratio"], data["custom_mutation_ratio"], data["finalized_generation"], data["threads"], data["double_tournament"], data["sl_model"], data["elite_selection"], data["eps"], data["eps_decrease"], data["gamma"], data["only_one_agent"])

training_type = 3
tg = 1
do_this = False
if do_this:
    print(f"brisca single 2 players")
    start_time_brisca = time.time()
    a = Supervised_training(training_type, total_games=tg, game_type=1, num_players=2, single_mode=True, only_assist=False, rivals_model_type=[1, 1, 1, 1], rivals_model_name=[None, None, None, None], generate_data=False, csv_filename='data/brisca/2j/20240519_225107_8000_partides_40_20_prepared_normalized.csv', csv_filename_2=None, save_prepared_data=False, save_filename='8000_partides_3_capes', do_training=True)
    a = None
    gc.collect()
    print("--- %s seconds ---" % (time.time() - start_time_brisca))

    print(f"tute single no assist 2 players")
    start_time_tute = time.time()
    a = Supervised_training(training_type, total_games=tg, game_type=2, num_players=2, single_mode=True, only_assist=False, rivals_model_type=[1, 1, 1, 1], rivals_model_name=[None, None, None, None], generate_data=False, csv_filename='data/tute/2j/20240519_231912_8000_partides_40_20_prepared_normalized.csv', csv_filename_2=None, save_prepared_data=False, save_filename='8000_partides_3_capes', do_training=True)
    a = None
    gc.collect()
    print("--- %s seconds ---" % (time.time() - start_time_tute))

    print(f"tute single assist 2 players")
    start_time_tute = time.time()
    a = Supervised_training(training_type, total_games=tg, game_type=2, num_players=2, single_mode=True, only_assist=True, rivals_model_type=[1, 1, 1, 1], rivals_model_name=[None, None, None, None], generate_data=False, csv_filename='data/tute_only_assist/2j/20240520_081404_8000_partides_40_20_prepared_normalized.csv', csv_filename_2=None, save_prepared_data=False, save_filename='8000_partides_3_capes', do_training=True)
    a = None
    gc.collect()
    print("--- %s seconds ---" % (time.time() - start_time_tute))

    print(f"brisca single 3 players")
    start_time_brisca = time.time()
    a = Supervised_training(training_type, total_games=tg, game_type=1, num_players=3, single_mode=True, only_assist=False, rivals_model_type=[1, 1, 1, 1], rivals_model_name=[None, None, None, None], generate_data=False, csv_filename='data/brisca/3j/20240520_085452_8000_partides_40_20_prepared_normalized.csv', csv_filename_2=None, save_prepared_data=False, save_filename='8000_partides_3_capes', do_training=True)
    a = None
    gc.collect()
    print("--- %s seconds ---" % (time.time() - start_time_brisca))

    print(f"tute single no assist 3 players")
    start_time_tute = time.time()
    a = Supervised_training(training_type, total_games=tg, game_type=2, num_players=3, single_mode=True, only_assist=False, rivals_model_type=[1, 1, 1, 1], rivals_model_name=[None, None, None, None], generate_data=False, csv_filename='data/tute/3j/20240520_090938_8000_partides_40_20_prepared_normalized.csv', csv_filename_2=None, save_prepared_data=False, save_filename='8000_partides_3_capes', do_training=True)
    a = None
    gc.collect()
    print("--- %s seconds ---" % (time.time() - start_time_tute))

    print(f"tute single assist 3 players")
    start_time_tute = time.time()
    a = Supervised_training(training_type, total_games=tg, game_type=2, num_players=3, single_mode=True, only_assist=True, rivals_model_type=[1, 1, 1, 1], rivals_model_name=[None, None, None, None], generate_data=False, csv_filename='data/tute_only_assist/3j/20240520_092927_8000_partides_40_20_prepared_normalized.csv', csv_filename_2=None, save_prepared_data=False, save_filename='8000_partides_3_capes', do_training=True)
    a = None
    gc.collect()
    print("--- %s seconds ---" % (time.time() - start_time_tute))

    print(f"brisca single 4 players")
    start_time_brisca = time.time()
    a = Supervised_training(training_type, total_games=tg, game_type=1, num_players=4, single_mode=True, only_assist=False, rivals_model_type=[1, 1, 1, 1], rivals_model_name=[None, None, None, None], generate_data=False, csv_filename='data/brisca/4j/20240520_094925_8000_partides_40_20_prepared_normalized.csv', csv_filename_2=None, save_prepared_data=False, save_filename='8000_partides_3_capes', do_training=True)
    a = None
    gc.collect()
    print("--- %s seconds ---" % (time.time() - start_time_brisca))

    print(f"tute single no assist 4 players")
    start_time_tute = time.time()
    a = Supervised_training(training_type, total_games=tg, game_type=2, num_players=4, single_mode=True, only_assist=False, rivals_model_type=[1, 1, 1, 1], rivals_model_name=[None, None, None, None], generate_data=False, csv_filename='data/tute/4j/20240520_100530_8000_partides_40_20_prepared_normalized.csv', csv_filename_2=None, save_prepared_data=False, save_filename='8000_partides_3_capes', do_training=True)
    a = None
    gc.collect()
    print("--- %s seconds ---" % (time.time() - start_time_tute))

    print(f"tute single assist 4 players")
    start_time_tute = time.time()
    a = Supervised_training(training_type, total_games=tg, game_type=2, num_players=4, single_mode=True, only_assist=True, rivals_model_type=[1, 1, 1, 1], rivals_model_name=[None, None, None, None], generate_data=False, csv_filename='data/tute_only_assist/4j/20240520_102617_8000_partides_40_20_prepared_normalized.csv', csv_filename_2=None, save_prepared_data=False, save_filename='8000_partides_3_capes', do_training=True)
    a = None
    gc.collect()
    print("--- %s seconds ---" % (time.time() - start_time_tute))

    print(f"brisca team 4 players")
    start_time_brisca = time.time()
    # Supervised_training(training_type, total_games=tg, game_type=1, num_players=i, single_mode=False, only_assist=False, rivals_model_type=[1, 1, 1, 1], rivals_model_name=[None, None, None, None], generate_data=True, csv_filename=None, prepare_data=True, train=True)
    # Supervised_training(training_type, total_games=tg, game_type=1, num_players=i, single_mode=False, only_assist=False, rivals_model_type=[1, 1, 1, 1], rivals_model_name=[None, None, None, None], generate_data=False, csv_filename='data/brisca/2j/20240401_215008.csv', prepare_data=True, train=True)
    a = Supervised_training(training_type, total_games=tg, game_type=1, num_players=4, single_mode=False, only_assist=False, rivals_model_type=[1, 1, 1, 1], rivals_model_name=[None, None, None, None], generate_data=False, csv_filename='data/brisca/4jt/20240520_104822_8000_partides_40_20_prepared_normalized.csv', csv_filename_2=None, save_prepared_data=True, save_filename='8000_partides_3_capes', do_training=True)
    a = None
    gc.collect()
    print("--- %s seconds ---" % (time.time() - start_time_brisca))

    print(f"tute team no assist 4 players")
    start_time_tute = time.time()
    a = Supervised_training(training_type, total_games=tg, game_type=2, num_players=4, single_mode=False, only_assist=False, rivals_model_type=[1, 1, 1, 1], rivals_model_name=[None, None, None, None], generate_data=False, csv_filename='data/tute/4jt/20240520_110521_8000_partides_40_20_prepared_normalized.csv', csv_filename_2=None, save_prepared_data=True, save_filename='8000_partides_3_capes', do_training=True)
    a = None
    gc.collect()
    print("--- %s seconds ---" % (time.time() - start_time_tute))

    print(f"tute team assist 4 players")
    start_time_tute = time.time()
    a = Supervised_training(training_type, total_games=tg, game_type=2, num_players=4, single_mode=False, only_assist=True, rivals_model_type=[1, 1, 1, 1], rivals_model_name=[None, None, None, None], generate_data=False, csv_filename='data/tute_only_assist/4jt/20240520_112634_8000_partides_40_20_prepared_normalized.csv', csv_filename_2=None, save_prepared_data=True, save_filename='8000_partides_3_capes', do_training=True)
    a = None
    gc.collect()
    print("--- %s seconds ---" % (time.time() - start_time_tute))

# TEST
# a = Genetic_training_rl(1, 10, 2, True, CUSTOM_RULES, False, False, 'test_4_generations', 200, 20, 4, 0, 5, 1, 0, True, 500, False, None, True, 0.05, 1e-7, 1.0, False)
# print("--- %s seconds ---" % (time.time() - start_time))


rules = {'can_change': False, 'last_tens': False, 'black_hand': False, 'hunt_the_three': False, 'only_assist': False}
# do_this = False
# if do_this:
episodes = 2000000
# a = Reinforcement_training(2, episodes, 2, False, rules, [f'rl_models/tute_only_assist/2j/0000_20240521_073000_3000000_partides_mc_multiple_key_eps_01_gamma_1_negative_points_only_one_agent', None, None, None], 0.1, 1e-7, 1.0, True, True)
a = None
gc.collect()
episodes = 1000000
# a = Reinforcement_training(2, episodes, 2, False, rules, [f'rl_models/tute_only_assist/2j/0000_20240521_073000_3000000_partides_mc_multiple_key_eps_01_gamma_1_negative_points_only_one_agent', None, None, None], 0.1, 1e-7, 1.0, True, True)
a = None
gc.collect()

episodes = 1000000
# a = Reinforcement_training(2, episodes, 3, False, rules, [f'rl_models/tute_only_assist/3j/0000_20240521_073000_3000000_partides_mc_multiple_key_eps_01_gamma_1_negative_points_only_one_agent', None, None, None], 0.1, 1e-7, 1.0, True, True)
a = None
gc.collect()
episodes = 2000000
a = Reinforcement_training(2, episodes, 3, False, rules, [f'rl_models/tute_only_assist/3j/0000_20240521_073000_3000000_partides_mc_multiple_key_eps_01_gamma_1_negative_points_only_one_agent', None, None, None], 0.1, 1e-7, 1.0, True, True)
a = None
gc.collect()

episodes = 1000000
a = Reinforcement_training(2, episodes, 4, False, rules, [f'rl_models/tute_only_assist/4j/0000_20240521_073000_3000000_partides_mc_multiple_key_eps_01_gamma_1_negative_points_only_one_agent', None, None, None], 0.1, 1e-7, 1.0, True, True)
a = None
gc.collect()
episodes = 2000000
a = Reinforcement_training(2, episodes, 4, False, rules, [f'rl_models/tute_only_assist/4j/0000_20240521_073000_3000000_partides_mc_multiple_key_eps_01_gamma_1_negative_points_only_one_agent', None, None, None], 0.1, 1e-7, 1.0, True, True)
a = None
gc.collect()

episodes = 1000000
a = Reinforcement_training(2, episodes, 4, False, rules, [f'rl_models/tute_only_assist/4jt/0000_20240521_073000_3000000_partides_mc_multiple_key_eps_01_gamma_1_negative_points_only_one_agent', None, None, None], 0.1, 1e-7, 1.0, True, True)
a = None
gc.collect()
episodes = 2000000
a = Reinforcement_training(2, episodes, 4, False, rules, [f'rl_models/tute_only_assist/4jt/0000_20240521_073000_3000000_partides_mc_multiple_key_eps_01_gamma_1_negative_points_only_one_agent', None, None, None], 0.1, 1e-7, 1.0, True, True)
a = None
gc.collect()

print("--- %s seconds ---" % (time.time() - start_time))
