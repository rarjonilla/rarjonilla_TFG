from typing import Dict

# CONSTANTS BACK END
NUM_CARDS_BRISCA: int = 3
NUM_CARDS_TUTE: int = 8

NUM_CARD_CHANGE_HIGH: int = 7
NUM_CARD_CHANGE_LOW: int = 2

# hunt the three és només vàlid per 2 jugadors
# only assist és només vàlid per al Tute
DEFAULT_RULES: Dict = {
    'can_change': True,
    'last_tens': True,
    'black_hand': True,
    'hunt_the_three': True,
    'only_assist': False,
}

SUITS: Dict = {
    1: 'ors',
    2: 'bastos',
    3: 'espases',
    4: 'copes',
}

CARDS: Dict = {
    1: {
        'card_num': 1,
        'card_value': 11,
        'training_pos': 10
    },
    2: {
        'card_num': 2,
        'card_value': 0,
        'training_pos': 1
    },
    3: {
        'card_num': 3,
        'card_value': 10,
        'training_pos': 9
    },
    4: {
        'card_num': 4,
        'card_value': 0,
        'training_pos': 2
    },
    5: {
        'card_num': 5,
        'card_value': 0,
        'training_pos': 3
    },
    6: {
        'card_num': 6,
        'card_value': 0,
        'training_pos': 4
    },
    7: {
        'card_num': 7,
        'card_value': 0,
        'training_pos': 5
    },
    8: {
        'card_num': 10,
        'card_value': 2,
        'training_pos': 6
    },
    9: {
        'card_num': 11,
        'card_value': 3,
        'training_pos': 7
    },
    10: {
        'card_num': 12,
        'card_value': 4,
        'training_pos': 8
    },
}

# CONSTANTS PLAYABLE
WINDOW_X_SIZE = 1500
WINDOW_Y_SIZE = 1000

CARD_X_SIZE = 105
CARD_Y_SIZE = 160

# Miliseconds
SLEEP_TIME = 1000
TIME_MOVING_CARD = 1
# Seconds
DEAL_SLEEP_TIME = 1.5
DEAL_INSTANT_SLEEP_TIME = 0.3
MOVING_TO_NEW_HAND_POS_TIME = 1.2
CHANGE_SLEEP_TIME = 2.5
PLAY_CARD_SLEEP_TIME = 2.5
PLAYER_PLAY_CARD_SLEEP_TIME = 1.5
SING_TIME = 3
AFTER_SING_TIME = 1.5
SCORE_TIME = 3

# Zona rectangular on es juguen les cartes
PLAYABLE_FRAME_POS_X = 255
PLAYABLE_FRAME_POS_Y = 385
PLAYABLE_FRAME_SIZE_X = 1000
PLAYABLE_FRAME_SIZE_Y = 220

PLAYABLE_CARD_INITIAL_POS_X_2_PLAYERS = 600
PLAYABLE_CARD_INITIAL_X_WIDTH_2_PLAYERS = 400
PLAYABLE_CARD_INITIAL_POS_X_3_PLAYERS = 550
PLAYABLE_CARD_INITIAL_X_WIDTH_3_PLAYERS = 250
PLAYABLE_CARD_INITIAL_POS_X_4_PLAYERS = 525
PLAYABLE_CARD_INITIAL_X_WIDTH_4_PLAYERS = 180
PLAYABLE_CARD_INITIAL_POS_Y = 415

# Zona rectangular on es mostra la informació
INFO_FRAME_POS_X = 255
INFO_FRAME_POS_Y = 90
INFO_FRAME_SIZE_X = 1000
INFO_FRAME_SIZE_Y = 120
INFO_FRAME_LABEL_SIZE_X = 138
INFO_FRAME_LABEL_SIZE_Y = 6

# Posicio de la baralla i la carta de triomf
DECK_POS_X = 275
DECK_POS_Y = 435
TRUMP_CARD_POS_X = 300
TRUMP_CARD_POS_Y = 465

# Mides de la zona de cartes per a cada jugador
HORIZONTAL_PLAYER_SIZE_X = 1000
VERTICAL_PLAYER_SIZE_Y = 920
HORIZONTAL_PLAYER_POS_X = 248
VERTICAL_PLAYER_POS_Y = 40

# Inici Posició a la pantalla de la zona de cartes per a cada jugador
PLAYER_0_POS_Y = 790
PLAYER_1_POS_X = 10
PLAYER_2_POS_Y = 210
PLAYER_3_POS_X = 1290

# Comptador de cartes
# Per a labels amb text, el size es compta per mida de caràcter "0"
# si s'indica 20 per l'eix "X", vol dir que tindrà la mida de 20 caràcters "0"
# si s'indica 20 per l'eix "Y", vol dir que tindrà la mida de 20 línies de caràcters "0"
DECK_COUNTER_SIZE_X = 4
DECK_COUNTER_SIZE_Y = 2
DECK_COUNTER_POS_X = 315
DECK_COUNTER_POS_Y = 395

# Puntuació de jugador
PLAYER_SCORE_SIZE_X = 200
PLAYER_SCORE_SIZE_Y = 80
PLAYER_SCORE_LABEL_SIZE_X = 30
PLAYER_SCORE_ROUND_LABEL_SIZE_X = 5

PLAYER_0_SCORE_POS_X = 255
PLAYER_0_SCORE_POS_Y = 10
PLAYER_1_SCORE_POS_X = 513
PLAYER_1_SCORE_POS_Y = 10
PLAYER_2_SCORE_POS_X = 771
PLAYER_2_SCORE_POS_Y = 10
PLAYER_3_SCORE_POS_X = 1029
PLAYER_3_SCORE_POS_Y = 10

# Posició botons
BTN_SIZE_X = 14
BTN_CHANGE_POS_X = 400
BTN_CHANGE_POS_Y = 670
BTN_SING_GOLD_POS_X = 530
BTN_SING_GOLD_POS_Y = 670
BTN_SING_COARSE_POS_X = 660
BTN_SING_COARSE_POS_Y = 670
BTN_SING_CUPS_POS_X = 790
BTN_SING_CUPS_POS_Y = 670
BTN_SING_SWORDS_POS_X = 920
BTN_SING_SWORDS_POS_Y = 670
BTN_NEW_GAME_POS_X = 1050
BTN_NEW_GAME_POS_Y = 670


# Pot ser RGB
BACKGROUND_COLOR = 'green'
