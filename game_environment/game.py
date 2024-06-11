import copy
import gc
from tqdm import tqdm
from typing import List, Optional, Dict, Tuple
from uuid import uuid1, UUID

from game_environment.player import Player
from game_environment.deck import Deck
from game_environment.card import Card

from constants import NUM_CARDS_BRISCA, NUM_CARDS_TUTE, SUITS

from game_environment.round import Round
from game_environment.score import Score
from configuration import PRINT_CONSOLE, RL_SAVE_AFTER_X_EPISODES
from training.game_state import Game_state
from training.player_state import Player_state

from keras import backend as K

from training.reinforcement.monte_carlo import Monte_carlo
from training.reinforcement.monte_carlo_multiple_key import Monte_carlo_multiple_state


class Game:
    """Classe Joc general"""

    def __init__(self, game_type: int, total_games: int, model_type: List[int], model_path: List[str], num_players: int, single_mode: bool, rules: Dict, training: bool, csv_filename: str, rl_eps: float = 0.05, rl_eps_decrease: float = 1e-7, rl_gamma: float = 1.0, rl_only_one_agent: bool = False, is_supervised_training: bool = False) -> None:
        # Total de partides a simular
        self._total_games: int = total_games
        # Tipus de joc (1-Brisca, 2- Tute)
        self.__game_type: int = game_type
        # Llista dels models per a cada jugador (1-Random, 7- GA, 8-SL heuristic i normalitzat, 10-RL múltiples estats)
        self.__model_type: List[int] = model_type
        # Llista dels path als models per a cada jugador (None- Random)
        self._model_path: List[str] = model_path

        # Llistat de jugadors
        self._players: List[Player] = []

        # Número de jugadors
        self._num_players: int = num_players

        # True- Es juga en solitari, False- Es juga en parelles
        self._single_mode: bool = single_mode

        # Regles que s'aplicaran durant la partida
        self._rules: Dict = rules

        # Màxima puntuació per si es guanya per caça del 3, mà negre o Tute de cavalls o reis
        max_score: int = 130 if self.is_rule_active('last_tens') else 120
        # Puntuació dels jugadors durant la simulació
        self._score: Score = Score(num_players, single_mode, max_score)

        # Es creen totes les cartes
        self._deck: Deck = Deck()

        # Indica el jugador que ha iniciat la partida
        self.__initial_player_id: int = -1
        # Indica el jugador que ha guanyat l'última ronda
        self._last_round_winner_id: int = self.__initial_player_id

        # Es crea la ronda inicial i un historic de rondes
        self._round: Round = Round(0, 0, False, rules, None)
        self._rounds: List[Round] = []

        # Indica si algun jugador ha guanyat per tute de cavalls o reis
        self._tutes: List[int] = []
        # Indica el jugador que ha guanyat per mà negre
        self._black_hand: int = -1
        # Indica el jugador que ha guanyat per caça al 3
        self._hunt_the_three: int = -1

        # Cartes que els jugadors tindran a la seva mà
        self._num_cards: int = NUM_CARDS_BRISCA if game_type == 1 else NUM_CARDS_TUTE

        # Per Entrenament RL -> True, un únic agent apren de totes les jugades, False -> hi ha "n" agents, un per cada jugador
        self._rl_only_one_agent: bool = rl_only_one_agent

        # Training and AI
        # Indica si s'està realitzant algun tipus d'entrenament
        self._training: bool = training
        # Nom del fitxer CSV a emmagatzemar per la generació del conjunt de dades de la SL
        self._csv_filename: str = csv_filename
        # Indica si es tracta d'un entrenament supervisat
        self._is_supervised_training: bool = is_supervised_training
        # Estat del joc, serveix per generar i emmagatzemar l'estat del joc de cada torn i jugador
        self._game_state: Game_state = Game_state(self._is_brisca(), self._single_mode, self._num_cards, self._num_players, self._rules, self.__model_type, is_supervised_training)

        # Si es tracta d'un entrenament RL amb un unic agent, es crear l'agent per pasar-lo com a parametre als jugadors (tots tindran la mateixa referència)
        if self._rl_only_one_agent:
            if self.__model_type[0] == 9:
                # RL amb un unic estat
                self._rl_agent: Monte_carlo = Monte_carlo(rl_eps, rl_eps_decrease, rl_gamma, model_type[0], model_path[0])
            elif self.__model_type[0] == 10:
                # RL amb múltiples estats
                self._rl_agent: Monte_carlo_multiple_state = Monte_carlo_multiple_state(rl_eps, rl_eps_decrease, rl_gamma, model_type[0], model_path[0], False, self._training)

        # Es creen els jugadors
        for id_player in range(0, num_players):
            if self._rl_only_one_agent:
                player: Player = Player(id_player, self.__model_type[id_player], self._model_path[id_player], rules, None, None, None, self._rl_agent, self._training)
            else:
                player: Player = Player(id_player, self.__model_type[id_player], self._model_path[id_player], rules, rl_eps, rl_eps_decrease, rl_gamma, None, self._training)
            self._players.append(player)

    def nullify_game(self):
        # Nul·lificació de la classe (per alliberar espai entre diferents entrenaments)
        for player in self._players:
            player.del_model()

        K.clear_session()

        del self._total_games
        del self.__game_type
        del self.__model_type
        del self._model_path
        del self._players
        del self._num_players
        del self._single_mode
        del self._rules
        del self._score
        del self._deck
        del self.__initial_player_id
        del self._last_round_winner_id
        del self._round
        del self._rounds
        del self._tutes
        del self._black_hand
        del self._hunt_the_three
        del self._num_cards
        del self._training
        del self._csv_filename
        del self._game_state

        gc.collect()

    # Public functions
    def get_player_wins_points(self, player_id: int) -> Tuple[int, int]:
        return self._score.get_player_wins(player_id), self._score.get_player_total_scores(player_id)

    def is_black_hand(self) -> bool:
        return self._black_hand != -1

    def is_hunt_the_three(self) -> bool:
        return self._hunt_the_three != -1

    def is_rule_active(self, rule_key: str) -> bool:
        return self._rules[rule_key]

    def is_tute(self) -> bool:
        return len(self._tutes) != 0

    # Private functions
    def __give_cards_to_players(self) -> None:
        # Es reparteix una carta a cada jugador
        for i in range(self._num_players):
            # El primer que agafa és el guanyador de l'última ronda i se segueix en sentit horari
            player_id: int = i + self._last_round_winner_id
            player_id %= self._num_players
            player = self._players[player_id]

            card: Card = self._deck.extract_card()
            player.hand_add_card_to_hand(card)

            # Training and AI
            if not self._deck.has_remaining_cards() and not self._deck.has_trump():
                # S'afegeix a l'estat del joc la carta coneguda d'un rival quan aquest ha robat la carta de triomf
                self._game_state.add_viewed_card(i, card)

    # TODO -> es pot eliminar
    def __reset_players_seen_cards(self) -> None:
        for player in self._players:
            player.reset_seen_cards()

    # Protected functions (Used in inheritance)
    def _game_results(self) -> None:
        # Es calculen els resultats finals de la partida segons com s'hagi finalitzat
        if self.is_black_hand() and self.is_tute():
            self._score.set_winners_by_black_hand_and_tute(self._black_hand, self._tutes)
        elif self.is_black_hand():
            self._score.set_winners_by_black_hand(self._black_hand)
        elif self.is_hunt_the_three():
            self._score.set_winners_by_hunt_the_three(self._hunt_the_three)
        else:
            self._score.set_winners_by_tute(self._tutes)

        if PRINT_CONSOLE:
            self._score.show_game_score()

        # Es finalitza la puntuació final
        self._score.finalize_score()

        # Training
        # S'indica a l'estat del joc la finalització de la partida per emmagatzemar les llistes de les jugades i resetejar-les
        self._game_state.finalize_game(self._score)

        if PRINT_CONSOLE:
            self._score.show_total_game_score()

        if PRINT_CONSOLE:
            for player in self._players:
                for singed_tute_suits in player.hand_singed_tute_suits():
                    print("Player ", player.get_id(), " singed tute of " + SUITS[singed_tute_suits])

        # Training RL
        # S'actualitza la política de l'agent RL (sí és únic només cal executar-la 1 cop, sinó cada jugador executa la seva)
        if self._training:
            if self._rl_only_one_agent:
                if self._players[0].is_model_type_rl():
                    self._players[0].rl_update_policy()
            else:
                for player in self._players:
                    if player.is_model_type_rl():
                        player.rl_update_policy()

    def _has_ended_game(self) -> bool:
        return not self._players[0].hand_has_cards()

    def _is_brisca(self) -> bool:
        return self.__game_type == 1

    def _new_game(self) -> None:
        # Es crea la baralla
        self._deck = Deck()

        # S'inicialitzen les variables
        self._black_hand = -1
        self._hunt_the_three = -1

        # Comença la partida el jugador a l'esquerra del que l'ha començat amb anterioritat
        self.__initial_player_id += 1
        self.__initial_player_id %= self._num_players
        self._last_round_winner_id = self.__initial_player_id

        for player in self._players:
            # Reiniciem la mà del jugador
            player.init_hand()

            # Si el model es RL, s'ha de iniciar un nou episodi
            if player.is_model_type_rl():
                player.rl_new_episode()

        # Training and AI
        # S'inicialitiza una nova partida per els estats del joc
        self._game_state.new_game()

        # Repartim cartes a cada jugador (1 a cadascun fins a arribar a les necessàries)
        for _ in range(0, self._num_cards):
            self.__give_cards_to_players()

        # S'extreu la carta de triomf
        self._deck.extract_trump_card()

        if PRINT_CONSOLE:
            print("trump card: " + str(self._deck.get_trump_card()))

        # Training and AI
        trump_card = self._deck.get_trump_card()
        if trump_card is None:
            raise AssertionError("Trump card is None")

        # S'indica la carta de triomf a l'estat del joc per aquesta partida
        self._game_state.change_trump_card(trump_card)

    def _next_round(self, first_round: bool = False) -> None:
        # Si la ronda es la primera no cal donar carta als jugadors
        # Amb 3 jugadors es dona el cas que, l'última carta (el triomf), no és de cap jugador
        # Aquesta es podrà intercanviar inclús a les últimes rondes, quan els jugadors ja no agafen cap carta més
        # Amb la condició self.deck.deck_size() > self.num_players es comprova aquesta condició, sense afectar a la resta de modes, indiferentment del nombre de jugadors
        if not first_round and self._deck.has_remaining_cards():
            self.__give_cards_to_players()

        # Training and AI
        # S'inicia nova ronda per els estats del joc
        self._game_state.new_round(first_round, self._deck.get_real_deck_size(), self._score.get_individual_scores())

        if PRINT_CONSOLE:
            print("carta de triomf", self._deck.get_trump_card())
            print("Mà dels jugadors: ")
            for player in self._players:
                player.show_hand()

        # Si algun jugador té els 4 reis o els 4 cavallers, guanya la partida automàticament (tute)
        self._tutes = []
        for player in self._players:
            has_tute, tute_kings = player.hand_has_tute()
            if has_tute:
                self._tutes.append(player.get_id())
            if self.is_rule_active('black_hand') and player.hand_has_black_hand(self._deck.get_trump_suit_id()):
                self._black_hand = player.get_id()

    def _player_turn(self, player: Player) -> Tuple[int, Optional[Card]]:
        # Training and AI
        # S'indica nou torn als estats del joc
        self._game_state.new_turn(player.get_id(), player.hand_get_cards_copy())

        # Aquests valors serveixen per comprovar quines són les cartes que el jugador pot jugar (a la Brisca és qualsevol, però al Tute està condicionat per les cartes que hi ha en joc)
        highest_suit_card: Optional[Card] = None if self._is_brisca() else self._round.highest_suit_card()
        deck_has_cards: bool = False if self._is_brisca() else self._deck.has_remaining_cards()
        highest_trump_played: Optional[Card] = None if self._is_brisca() else self._round.highest_trump_played()

        # Es fa una còpia de les cartes que pot jugar (es modificarà més endavant i no volem que es modifiqui la mà original)
        copied_playable_hand: List[Card] = copy.deepcopy(player.hand_get_playable_cards(self._deck.get_trump_suit_id(), highest_suit_card, deck_has_cards, highest_trump_played))

        # S'agafa l'estat del joc per aquest torn
        player_state: Optional[Player_state] = None
        if not player.is_model_type_random():
            player_state = self._game_state.get_player_state(player.get_id())

        # Es calcula l'acció que el jugador durà a terme
        # 0 = canvi de carta, 1-8 = posició de la carta jugada, Carta que juga (ja s'ha eliminat de la seva mà). Si és None és que intercanvia la carta de triomf
        card_position, card_or_change = player.get_next_action(self._deck.get_trump_card() is not None, self._deck.is_high_trump(), self._deck.get_trump_suit_id(), highest_suit_card, deck_has_cards, highest_trump_played, player_state)

        if card_or_change is None:
            # Acció d'intercanvi de la carta de triomg
            if PRINT_CONSOLE:
                print("Acció: intercanviar carta")

            # S'agafa la posició i carta de la seva mà
            card_to_change_position, card_to_change = player.hand_card_to_change(self._deck.is_high_trump(), self._deck.get_trump_suit_id())
            card_position = card_to_change_position

            # Es realitza l'intercanvi
            trump_card = self._deck.get_trump_card()
            if trump_card is None or card_to_change is None:
                raise AssertionError("Card or Trump card is None")
            else:
                # Training and AI
                # S'indica carta coneguda del rival (la carta de triomf)
                self._game_state.add_viewed_card(player.get_id(), trump_card)

                if PRINT_CONSOLE:
                    print(card_to_change)
                    print("per")
                    print(trump_card)

                # S'afegeix la carta de triomf a la mà del jugador
                player.hand_change_card(card_to_change_position, trump_card)

                # S'actualitza la carta de triomf
                self._deck.change_trump_card(card_to_change)

                if PRINT_CONSOLE:
                    player.show_hand()

                # Comprovar mà negre (el jugador pot guanyar amb aquest intercanvi)
                if self.is_rule_active('black_hand') and player.hand_has_black_hand(self._deck.get_trump_suit_id()):
                    self._black_hand = player.get_id()
        else:
            # Acció de jugar una carta
            self._round.played_card(card_or_change)

            # Training and AI
            # S'elimina la carta de les vistes dels rivals (si es que estava visible) de l'estat del joc
            self._game_state.remove_viewed_card(player.get_id(), card_or_change)
            # S'afegeix una carta jugada durant la partida a l'estat del joc
            self._game_state.add_played_card(card_or_change)

            # Es calcula l'heuristic per la SL de la jugada
            self._game_state.heuristics(player.get_id(), card_or_change, copied_playable_hand)

        return card_position, card_or_change

    def _round_results(self) -> None:
        # Càlcul del resultat de la ronda
        if not self.is_rule_active('black_hand') or not self.is_black_hand():
            # No s'ha guanyat per mà negre
            # es calcula el resultat i es retorna si el jugador que ha guanyat punts per cantar les 20 o les 40
            singed_tute: bool = self._round.calc_round_results()
            # S'indica el guanyador de l'última ronda
            self._last_round_winner_id = self._round.get_round_winner()

            # Training RL -> Afegir recompensa i guardar a la memòria de l'agent
            if self._training:
                for player in self._players:
                    if player.is_model_type_rl():
                        if player.get_id() == self._last_round_winner_id or (not self._single_mode and player.get_id() % 2 == self._last_round_winner_id % 2):
                            player.rl_set_reward(self._round.get_round_points())
                        else:
                            # player.rl_set_reward(0)
                            player.rl_set_reward(-self._round.get_round_points())
                        if player.is_model_type_rl():
                            player.rl_add_memory()

            singed_tute_suit: int = 0
            if singed_tute:
                # Si ha cantat, s'indica al jugador que ja no pot tornar a cantar d'aquest pal
                singed_tute_suit = self._round.get_singed_suit()
                self._players[self._last_round_winner_id].hand_add_singed_tute_suit(singed_tute_suit)

            if PRINT_CONSOLE:
                self._round.show_round()

            # S'afegeix la ronda a l'historial
            self._rounds.append(self._round)
            # S'actualitza la puntuació de la partida
            self._score.add_score(self._last_round_winner_id, self._round.get_round_points())

            if PRINT_CONSOLE:
                print("")

            # Training
            # S'indica a l'estat del joc que ha finalitzat la ronda
            self._game_state.finalize_round(self._round)


class Non_playable_game(Game):
    """Classe Joc no jugable (simulacions)"""
    def __init__(self, game_type: int, total_games: int, model_type: List[int], model_path: List[Optional[str]], num_players: int, single_mode: bool, rules: Dict, training: bool, csv_filename: str, rl_eps: float = 0.05, rl_eps_decrease: float = 1e-7, rl_gamma: float = 1.0, rl_only_one_agent: bool = False, is_supervised_training: bool = False) -> None:
        # Es crida el constructor general
        super().__init__(game_type, total_games, model_type, model_path, num_players, single_mode, rules, training, csv_filename, rl_eps, rl_eps_decrease, rl_gamma, rl_only_one_agent, is_supervised_training)
        # Es comença la simulació
        self.__start_game()

    # Private functions
    def __next_round(self, first_round: bool = False) -> None:
        # S'inicia la següent ronda
        self._next_round(first_round)

        if not self.is_tute() and not self.is_black_hand():
            # No s'ha guanyat per Tute ni per mà negra
            # Variable per emmagatzemar els cants dels jugadors a l'inici de la ronda
            sing_declarations: Optional[List[Optional[int]]] = None if self._is_brisca() else []

            if not self._is_brisca() and sing_declarations is not None:
                # Algú pot cantar
                # Calculem els cantics de Tute de cada jugador
                for player in self._players:
                    # S'agafa l'estat del joc del torn
                    player_state = self._game_state.get_player_state(player.get_id())
                    if player_state is None:
                        # És la primera ronda de la partida, encara no s'ha inicialitzat el nou torn de l'estat del joc
                        self._game_state.new_turn(player.get_id(), player.hand_get_cards_copy())
                        # S'agafa l'estat del joc del torn
                        player_state = self._game_state.get_player_state(player.get_id())

                    # Es calcula l'acció de triar cant per el jugador
                    sing_declaration: Optional[int] = player.get_next_action_sing_declarations(self._deck.get_trump_suit_id(), player_state)
                    sing_declarations.append(sing_declaration)

                    # Training and AI
                    if sing_declaration is not None:
                        # S'inicialitza un nou torn i s'indica l'acció triada (quin cant vol fer dels que pot)
                        self._game_state.new_turn(player.get_id(), player.hand_get_cards_copy())
                        self._game_state.set_action(player.get_id(), sing_declaration, 2)
                        self._game_state.add_current_to_round(player.get_id())
                        # S'inicialitza un nou torn
                        self._game_state.new_turn(player.get_id(), player.hand_get_cards_copy())

                        # S'agafen les cartes del cant per afegir-les a les cartes conegudes del rival a l'estat del joc
                        sing_cards_pos: List[int] = player.hand_get_sing_cards_position(sing_declaration)
                        for sing_card_pos in sing_cards_pos:
                            card_pos, card = player.hand_get_card_in_position_no_remove(sing_card_pos)
                            self._game_state.add_viewed_card(player.get_id(), card)

            # Training and AI
            # S'indica a l'estat del joc els cants dels jugadors
            self._game_state.set_sing_declarations(sing_declarations)

            # S'inicialitza una nova ronda
            self._round: Round = Round(self._last_round_winner_id, self._deck.get_trump_suit_id(), self._players[0].hand_has_one_left_card(), self._rules, sing_declarations)

            player_turn = self._last_round_winner_id
            for i in range(self._num_players):
                # Es realitza el torn dels jugadors en ordre
                player = self._players[player_turn]
                self.__player_turn(player)

                # Ha aconseguit mà negra fent un intercanvi
                if self._black_hand != -1:
                    break
                else:
                    player_turn += 1
                    player_turn %= self._num_players

            if not self.is_black_hand():
                # Ningú ha guanyat per mà negra
                if self.is_rule_active('hunt_the_three') and self._num_players == 2:
                    # Es comprova si algú ha caçat el 3
                    self._hunt_the_three = self._round.hunt_the_three(not self._players[0].hand_has_cards())

                # Es calcula el resultat de la ronda
                self._round_results()

        if not self._has_ended_game() and not self.is_tute() and not self.is_black_hand() and not self.is_hunt_the_three():
            # Si no ha finalitzat el joc, es comença una ronda nova
            self.__next_round(first_round=False)

    def __player_turn(self, player: Player) -> Tuple[int, Optional[Card]]:
        # Es calcula el torn del jugador
        card_position, card_or_change = self._player_turn(player)

        # Training and AI
        if card_or_change is not None:
            # Accions -> 0 = intercanvi, 1=carta posicio 0, ...
            # type -> 0=canvi, 1=carta, 2=cant
            # S'indica a l'estat del joc l'acció del jugador
            self._game_state.set_action(player.get_id(), card_or_change.get_training_idx(), 1)
        else:
            # S'indica a l'estat del joc l'acció d'intercanvi
            self._game_state.set_action(player.get_id(), 0, 0)

        # Training and AI
        # S'afegeix l'estat i acció a la llista del jugador
        self._game_state.add_current_to_round(player.get_id())

        if card_or_change is None and (not self.is_rule_active('black_hand') or not player.hand_has_black_hand(self._deck.get_trump_suit_id())):
            # Si s'ha fet intercanvi de carta i no guanya per mà negra, ha de seguir jugant el mateix jugador
            self.__player_turn(player)

        return card_position, card_or_change

    def __start_game(self) -> None:
        # Inici de les simulacions
        game_num: int = 1

        for game_num in tqdm(range(self._total_games)):
            # inicialització de la partida
            self._score.reset_last_winners()
            self._new_game()
            self.__next_round(first_round=True)
            self._game_results()

            # Training RL
            # Es guarda el model de RL cada X simulacions
            if self._training and game_num % RL_SAVE_AFTER_X_EPISODES == 0 and game_num != self._total_games and game_num != 0:
                # Es guarda només un cop si és un unic agent, o es guarda per a cada jugador
                if self._rl_only_one_agent:
                    self._rl_agent.save_model()
                    print(self._model_path[0], " saved game ", game_num)
                else:
                    for player in self._players:
                        if player.is_model_type_rl():
                            player.rl_save_model()
                            print(self._model_path[player.get_id()], " saved game ", game_num)

            game_num += 1

            if PRINT_CONSOLE:
                print("game_environment ", game_num, " of ", self._total_games, end="\r")

        # Es mostra el resultat de les simulacions
        self._score.show_total_game_score()

        # Training SL
        # Emmagatzemar csv amb totes les dades (conjunt per la SL)
        if self._training and self._csv_filename is not None:
            self._game_state.save_csv(self._csv_filename)

        # Training RL
        # Es guarda el model de RL al finalitzar la simulació
        if self._training:
            # Es guarda només un cop si és un unic agent, o es guarda per a cada jugador
            if self._rl_only_one_agent:
                self._rl_agent.save_model()
                print(self._model_path[0], " saved game ", game_num)
            else:
                for player in self._players:
                    if player.is_model_type_rl():
                        player.rl_save_model()
                        print(self._model_path[player.get_id()], " saved")

    def score_get_total_scores(self) -> List[int]:
        return self._score.get_total_scores()

    def score_get_wins(self) -> List[int]:
        return self._score.get_wins()


class Playable_game(Game):
    """Classe del joc (GUI)"""
    def __init__(self, game_type: int, total_games: int, model_type: List[int], model_path: List[Optional[str]], num_players: int, single_mode: bool, rules: Dict, training: bool, csv_filename: str, human_player: bool, is_supervised_training: bool = False) -> None:
        # Es crida el constructor del joc general
        super().__init__(game_type, total_games, model_type, model_path, num_players, single_mode, rules, training, csv_filename, is_supervised_training=is_supervised_training)

        # Es guarda si el jugador 0 és un humà
        self.__human_player = human_player

    # Getters i Setters
    def get_black_hand(self) -> int:
        return self._black_hand

    def get_hunt_the_three(self) -> int:
        return self._hunt_the_three

    def get_last_round(self) -> Round:
        return self._rounds[-1]

    def get_last_round_winner_id(self) -> int:
        return self._last_round_winner_id

    def get_num_cards(self) -> int:
        return self._num_cards

    def get_player_by_position(self, player_position: int) -> Player:
        return self._players[player_position]

    def get_tutes(self) -> List[int]:
        return self._tutes

    def is_single_mode(self) -> int:
        return self._single_mode

    def set_black_hand(self, black_hand: int) -> None:
        self._black_hand = black_hand

    # Functions
    # retorna si s'ha de continuar jugant
    def continue_game(self) -> bool:
        return not self._has_ended_game() and not self.is_tute() and not self.is_black_hand() and not self.is_hunt_the_three()

    def get_player_turn(self, round_turn_idx: int) -> Player:
        player_turn = (self._last_round_winner_id + round_turn_idx) % self._num_players
        return self._players[player_turn]

    def finalize_game(self) -> None:
        self._game_results()

    def finalize_round(self) -> None:
        if self.is_rule_active('hunt_the_three') and self._num_players == 2 and not self.is_black_hand():
            # Es comprova si s'ha caçat el 3
            self._hunt_the_three = self._round.hunt_the_three(not self._players[0].hand_has_cards())

        # Es calcula el resultat de la ronda
        self._round_results()

    def is_brisca(self) -> bool:
        return self._is_brisca()

    def new_game(self) -> None:
        self._new_game()

    def next_round(self, first_round: bool = False) -> None:
        # Inicialització d'una nova ronda
        self._next_round(first_round)

        if not self.is_tute() and not self.is_black_hand():
            # Ningú guanya per Tute ni per mà negra
            sing_declarations: Optional[List[Optional[int]]] = None if self.is_brisca() else []

            if not self.is_brisca() and sing_declarations is not None:
                # Algú pot cantar
                # Calculem els cantics de Tute de cada jugador
                for player in self._players:
                    # Només si és un jugador no humà
                    if player.get_id() != 0 or not self.__human_player:
                        player_state: Optional[Player_state] = None
                        if not player.is_model_type_random():
                            # S'agafa l'estat del joc
                            player_state = self._game_state.get_player_state(player.get_id())

                        # Es calcula l'acció de cant
                        sing_declaration: Optional[int] = player.get_next_action_sing_declarations(self._deck.get_trump_suit_id(), player_state)
                        sing_declarations.append(sing_declaration)
                    else:
                        # Cas del jugador humà
                        # Automatic si en té només 1 o té "les 40")
                        # Haurà de triar un dels cants en cas de diversos "les 20"
                        sing_suits_ids: List[int] = player.hand_sing_suits_in_hand()

                        if self._deck.get_trump_suit_id() in sing_suits_ids:
                            # Si el jugador té les 40, se li escollirà automàticament
                            sing_declarations.append(self._deck.get_trump_suit_id())
                        elif len(sing_suits_ids) == 1:
                            # Si el jugador només té un cant, se li escollirà automàticament
                            sing_declarations.append(sing_suits_ids[0])
                        else:
                            # Si en te 2 haurà de triar amb els botons
                            sing_declarations.append(None)

            # S'inicialitza la ronda
            self._round: Round = Round(self._last_round_winner_id, self._deck.get_trump_suit_id(), self._players[0].hand_has_one_left_card(), self._rules, sing_declarations)

    def player_turn(self, player: Player) -> Tuple[int, Optional[Card]]:
        return self._player_turn(player)

    # Deck Getters
    def deck_change_trump_card(self, card: Card) -> None:
        return self._deck.change_trump_card(card)

    def deck_get_deck_size(self) -> int:
        return self._deck.get_deck_size()

    def deck_get_trump_card(self) -> Optional[Card]:
        return self._deck.get_trump_card()

    def deck_get_trump_suit_id(self) -> int:
        return self._deck.get_trump_suit_id()

    def deck_has_remaining_cards(self) -> bool:
        return self._deck.has_remaining_cards()

    def deck_is_high_trump(self) -> bool:
        return self._deck.is_high_trump()

    # Game State Getters
    def game_state_add_current_to_round(self, player_id: int) -> None:
        self._game_state.add_current_to_round(player_id)

    def game_state_add_played_card(self, card: Card) -> None:
        self._game_state.add_played_card(card)

    def game_state_change_trump_card(self) -> None:
        card: Optional[Card] = self._deck.get_trump_card()
        if card is None:
            raise AssertionError("Card is None")
        self._game_state.change_trump_card(card)

    def game_state_heuristics(self, player_id: int, played_card: Card, playable_hand: List[Card]) -> None:
        self._game_state.heuristics(player_id, played_card, playable_hand)

    def game_state_new_turn(self, player_id: int) -> None:
        self._game_state.new_turn(player_id, self.get_player_by_position(player_id).hand_get_cards_copy())

    def game_state_remove_viewed_card(self, player_id: int, card: Card) -> None:
        self._game_state.remove_viewed_card(player_id, card)

    def game_state_set_action(self, player_id: int, action: int, type: int) -> None:
        self._game_state.set_action(player_id, action, type)

    def game_state_set_sing_declarations(self) -> None:
        # S'indica les cartes conegudes corresponents als cants dels jugador a l'estat del joc
        sing_declarations = self.round_get_sing_declarations()
        self._game_state.set_sing_declarations(sing_declarations)

        if sing_declarations is not None:
            for i, sing_declaration in enumerate(sing_declarations):
                if sing_declaration is not None:
                    player = self.get_player_by_position(i)
                    sing_cards_pos: List[int] = player.hand_get_sing_cards_position(sing_declaration)
                    for sing_card_pos in sing_cards_pos:
                        card_pos, card = player.hand_get_card_in_position_no_remove(sing_card_pos)
                        self._game_state.add_viewed_card(player.get_id(), card)

    # Player Getters
    def player_hand_get_playable_cards(self, player_id: int) -> List[Card]:
        # Es retorna les cartes jugables del jugador
        highest_suit_card: Optional[Card] = None if self._is_brisca() else self._round.highest_suit_card()
        deck_has_cards: bool = False if self._is_brisca() else self._deck.has_remaining_cards()
        highest_trump_played: Optional[Card] = None if self._is_brisca() else self._round.highest_trump_played()

        return self.get_player_by_position(player_id).hand_get_playable_cards(self._deck.get_trump_suit_id(), highest_suit_card, deck_has_cards, highest_trump_played)

    # Round Getters
    def round_get_sing_declarations(self) -> Optional[List[Optional[int]]]:
        return self._round.get_sing_declarations()

    def round_get_singed_suit(self) -> int:
        return self._round.get_singed_suit()

    def round_has_singed(self) -> bool:
        return self._round.has_singed()

    def round_highest_suit_card(self) -> Optional[Card]:
        return self._round.highest_suit_card()

    def round_highest_trump_played(self) -> Optional[Card]:
        return self._round.highest_trump_played()

    def round_is_sing_suit(self) -> bool:
        return self._round.is_sing_suit()

    def round_played_card(self, card: Card) -> None:
        return self._round.played_card(card)

    # Score Getters
    def score_get_history_scores(self) -> List[List[int]]:
        return self._score.get_history_scores()

    def score_get_individual_scores(self) -> List[int]:
        return self._score.get_individual_scores()

    def score_get_last_winners(self) -> List[int]:
        return self._score.get_last_winners()

    def score_get_total_scores(self) -> List[int]:
        return self._score.get_total_scores()

    def score_get_wins(self) -> List[int]:
        return self._score.get_wins()

    def score_reset_last_winners(self) -> None:
        return self._score.reset_last_winners()
