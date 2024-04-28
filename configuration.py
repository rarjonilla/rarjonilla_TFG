# TOTAL_GAMES: El total de partides a simular
from typing import Dict, List, Optional

USE_GPU = True

# TOTAL_GAMES: Total de partides a simular (només no jugable)
TOTAL_GAMES: int = 1

# NUM_PLAYERS: Total de jugadors (entre 2 i 4)
NUM_PLAYERS: int = 3

# SINGLE_MODE: modalitat de joc individual o en parelles (només per a 4 jugadors) -> False=Es juga en parelles, True=Cada jugador juga individualment
SINGLE_MODE: bool = True

# GAME_TYPE: El tipus de joc ->  1=Brisca, 2=Tute
GAME_TYPE: int = 1

# MODEL_TYPE: El model que ha de jugar per a cada player
# 1=Random mode
# 2=Supervised NN Round points (hds, rp, hsp)
# 3=Supervised NN Win or Lose (hds, wl, hsp)
# 4=Supervised NN Round points (nds, rp, hsp)
# 5=Supervised NN Win or Lose (nds, wl, hsp)
# 6=Supervised NN Round points + heuristics (nds, wl, hsp)
# X=Genetic NN
# X=Reinforcement NN
# MODEL_TYPE: List[int] = [1, 1, 1, 1]
MODEL_TYPE: List[int] = [1, 6, 6, 6]
# MODEL_TYPE: List[int] = [3, 3, 1, 1]
# MODEL_TYPE: List[int] = [2, 2, 1, 1]
# MODEL_TYPE_TEST: List[int] = [1, 1, 1, 1]
# MODEL_TYPE_TEST: List[int] = [1, 4, 1, 1]
# MODEL_TYPE_TEST: List[int] = [1, 5, 1, 1]
# MODEL_TYPE_TEST: List[int] = [1, 6, 1, 1]
# VS
MODEL_TYPE_TEST: List[int] = [6, 6, 6, 6]

# None per aleatori
# MODEL_PATH: List[Optional[str]] = [None, None, None, None]
MODEL_PATH: List[Optional[str]] = [None, 'sl_models/brisca/3j/sl_heu_test_brisca_20240408_222040.h5', 'sl_models/brisca/3j/sl_heu_test_brisca_20240408_222040.h5', None]
# MODEL_PATH: List[Optional[str]] = ['sl_models/brisca/2j/st_hds_wl_20240330_190136.h5', 'sl_models/brisca/2j/st_hds_wl_20240330_190136.h5', None, None]
# MODEL_PATH: List[Optional[str]] = ['sl_models/brisca/2j/st_hds_rp_hsp_20240330_190136.h5', 'sl_models/brisca/2j/st_hds_rp_hsp_20240330_190136.h5', None, None]
# MODEL_PATH_TEST: List[Optional[str]] = [None, None, None, None]
# MODEL_PATH_TEST: List[Optional[str]] = [None, 'sl_models/brisca/2j/sl_t2_nds_rp_hsp_t2_20240405_141958_concat.h5', None, None]
# MODEL_PATH_TEST: List[Optional[str]] = [None, 'sl_models/brisca/2j/sl_t2_nds_wl_t2_20240405_141958_concat.h5', None, None]
# MODEL_PATH_TEST: List[Optional[str]] = [None, 'sl_models/brisca/2j/sl_t2_nds_heu_t2_20240405_141958_concat.h5', None, None]
# VS
MODEL_PATH_TEST: List[Optional[str]] = ['sl_models/brisca/2j/sl_t2_nds_heu_t2_20240405_141958_concat.h5', 'sl_models/brisca/2j/st_nds_heu_20240404_232428.h5', None, None]


# Configuració de les regles -> True=Activada, False=No activada
CUSTOM_RULES: Dict = {
    'can_change': False,
    'last_tens': False,
    'black_hand': False,
    'hunt_the_three': False,
    'only_assist': False,
}

# APPLY_CUSTOM_RULES: Regles aplicades -> False=Regles definides per defecte, True=Definides just a sobre
APPLY_CUSTOM_RULES: bool = True

# PRINT_CONSOLE: Mostrar missatges per consola -> True=Es mostren missatges de seguiment, False=Només es mostra missatge de finalització de partida
PRINT_CONSOLE: bool = False

# SHOW_RIVAL_CARDS: Mostrar o no les cartes del rival (només versió GUI) -> True=Es mostren les cartes de tots els jugadors, False=Només es mostren les cartes del jugador humà
SHOW_RIVAL_CARDS: bool = True

# INSTANT_DEAL: Com es reparteixen les cartes (només versió GUI) -> True=Sense animació (simulacions més ràpides). False=Amb animació (simulacions més lentes)
INSTANT_DEAL: bool = True

# HUMAN_PLAYER: Jugador humà (només versió GUI) -> True=El jugador 0 és un humà, False=Tots els jugadors són IA
# HUMAN_PLAYER: bool = True
