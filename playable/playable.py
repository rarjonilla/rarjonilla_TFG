from __future__ import annotations

import copy
from tkinter import *
from typing import Dict, List, Optional

from game_environment.card import Card
from constants import WINDOW_X_SIZE, WINDOW_Y_SIZE, SLEEP_TIME, PLAYABLE_FRAME_POS_Y, \
    PLAYABLE_FRAME_POS_X, PLAYABLE_FRAME_SIZE_Y, PLAYABLE_FRAME_SIZE_X, DECK_POS_X, DECK_POS_Y, \
    TRUMP_CARD_POS_X, TRUMP_CARD_POS_Y, DEAL_SLEEP_TIME, CHANGE_SLEEP_TIME, SING_TIME, \
    MOVING_TO_NEW_HAND_POS_TIME, SUITS, PLAYER_PLAY_CARD_SLEEP_TIME, DEAL_INSTANT_SLEEP_TIME, AFTER_SING_TIME
from game_environment.game import Playable_game
from playable.playable_buttons import Playable_buttons
from playable.playable_card import Playable_card
from playable.playable_deck_counter import Playable_deck_counter
from playable.playable_frames import Playable_frames
from playable.playable_utility_functions import tksleep, get_card_path, raise_frame_or_label, calc_played_card_position, \
    calc_hand_card_position, calc_dragged_card_position
from game_environment.player import Player
from configuration import SHOW_RIVAL_CARDS, INSTANT_DEAL

class Playable:
    """Classe Playable"""
    def __init__(self, game_type: int, human_player: bool, total_games: int, model_type: List[int], model_path: List[str], num_players: int, single_mode: bool, rules: Dict, training: bool, csv_filename: str) -> None:
        # Nombre de jugadors
        self.__num_players: int = num_players
        # Indica si juga un humà
        self.__human_player: bool = human_player

        # Variables del joc
        self.__cards_played: int = 0
        self.__game_has_ended: bool = False
        self.__continue_dealing = True
        self.__first_round: bool = True
        self.__card_dragged_position: int = -1
        self.__round_turn_idx: int = 0
        self.__last_initial_player: int = 0

        # Llista dels labels de les cartes
        self.__player_cards_lbl: List[List[Playable_card]] = []

        # Array de les posicions de les cartes jugades (per fer els moviments de les cartes al repartir)
        self.__played_cards_position: List[Optional[int]] = [None] * num_players

        # Inicialització de les cartes jugades
        for i in range(0, num_players):
            self.__player_cards_lbl.append([])

        # Crear Pantalla, frames i iniciar partida
        self.__root: Tk = Tk()
        self.__create_window()
        self.__frames: Playable_frames = Playable_frames(self.__root, num_players)

        # Labels del deck i triomf
        self.__lbl_deck: Optional[Playable_card] = Playable_card(self.__root, self.__frames.frm_board, get_card_path(None), None, None, None, DECK_POS_X, DECK_POS_Y, True)
        self.__lbl_trump_card: Optional[Playable_card] = None

        # S'inicialitza el joc (GUI)
        self.__game = Playable_game(game_type, total_games, model_type, model_path, num_players, single_mode, rules, training, csv_filename, human_player)
        # Definició dels botons
        self.__buttons: Playable_buttons = Playable_buttons(self.__frames.frm_board, self.__game.is_brisca(), rules['can_change'], self.__new_game, self.__choose_change_card, self.__choose_sing_suit, self.__human_player)
        # Inici de la partida
        self.__game.new_game()

        # +1 perquè la carta de triomf no es comptabilitza dins del deck
        self.__deck_counter = Playable_deck_counter(self.__frames.frm_board, self.__game.deck_get_deck_size() + 1)

        # Després de 2 segons, començar a repartir les cartes
        self.__root.after(SLEEP_TIME, self.__deal_cards, self.__game.get_num_cards(), True)

        self.__root.mainloop()

    # Window
    def __create_window(self) -> None:
        # Mida de la finestra
        self.__root.geometry(f'{WINDOW_X_SIZE}x{WINDOW_Y_SIZE}')
        # La finestra i les posicions estan fixades, no es permet modificar-ne la mida
        self.__root.resizable(False, False)

    #     def create_menu_bar(self):
    #         menubar = Menu(self.root)
    #         self.root.config(menu=menubar)  # Lo asignamos a la base
    #
    #         file_menu = Menu(menubar, tearoff=0)
    #         file_menu.add_command(label="Frame 1", command=lambda: self.raise_frame(self.frames.frm_board))
    #         # file_menu.add_command(label="Frame 2", command=lambda: self.raise_frame(self.f2))
    #         # file_menu.add_command(label="Frame 3", command=lambda: self.raise_frame(self.f3))
    #         # file_menu.add_command(label="Frame 4", command=lambda: self.raise_frame(self.f4))
    #         file_menu.add_command(label="Clear Frame 1", command=lambda: self.clear_frame(self.frames.frm_board))
    #         file_menu.add_separator()
    #         file_menu.add_command(label="Salir", command=self.root.quit)
    #
    #         edit_menu = Menu(menubar, tearoff=0)
    #         help_menu = Menu(menubar, tearoff=0)
    #
    #         menubar.add_cascade(label="Archivo", menu=file_menu)
    #         menubar.add_cascade(label="Editar", menu=edit_menu)
    #         menubar.add_cascade(label="Ayuda", menu=help_menu)

    # Game
    def __deal_cards(self, total_cards_to_deal: int, deal_trump: bool = False) -> None:
        # Primer jugador al que se li reparteix carta
        deal_player_id = self.__game.get_last_round_winner_id()

        # Es repateixen totes les cartes (entre 1 i 8 segons si és inici de joc o inici de ronda)
        for card_pos in range(0, total_cards_to_deal):
            for round_turn_idx in range(0, self.__num_players):
                player: Player = self.__game.get_player_turn(round_turn_idx)
                # Es comprova si és la carta de triomd
                is_trump_card = True if not self.__game.deck_has_remaining_cards() and round_turn_idx == (self.__num_players - 1) else False

                if self.__played_cards_position[player.get_id()] is None:
                    # Es reparteixen les cartes a l'inici de la partida, no cal fer moviments de la resta de cartes
                    card_pos, card = player.hand_get_card_in_position_no_remove(card_pos)
                    self.__deal_card(card, player.get_id(), card_pos, is_trump_card)
                else:
                    # La llista de cartes a la mà afegeix al final, per tant, sempre haig de triar la última carta de la seva mà, independentment de la posició jugada en pantalla
                    # Si era l'ultima carta del deck, podem eliminar la imatge del deck un cop repartida aquesta carta al penúltim jugador
                    if round_turn_idx == self.__num_players - 2:
                        if not self.__game.deck_has_remaining_cards() and self.__lbl_deck is not None:
                            self.__lbl_deck.destroy_label()
                            self.__lbl_deck = None

                    card_pos, card = player.hand_get_card_in_position_no_remove(self.__game.get_num_cards() - 1)
                    self.__deal_card(card, player.get_id(), self.__game.get_num_cards() - 1, is_trump_card)

                deal_player_id += 1
                deal_player_id %= self.__num_players

        self.__played_cards_position = [None] * self.__num_players

        if deal_trump:
            trump_card = self.__game.deck_get_trump_card()
            self.__lbl_trump_card = Playable_card(self.__root, self.__frames.frm_board, get_card_path(trump_card), None, None, None, TRUMP_CARD_POS_X, TRUMP_CARD_POS_Y, False)
            if self.__lbl_deck is not None:
                raise_frame_or_label(self.__lbl_deck.get_label())

            self.__next_round()
        total_cartes = self.__game.deck_get_deck_size() + 1 if self.__game.deck_has_remaining_cards() else 0
        self.__deck_counter.update_counter(total_cartes)

    def __end_game(self) -> None:
        self.__game_has_ended = True

        self.__game.finalize_game()

        # Comprovar si ha guanyat per hunt_the_three, black_hand, tute de cavalls / reis o black_hand i tute

        # Es pot donar el cas 1 i 2 alhora (tute de cavalls per un jugador i mà negre per un altre jugador)
        text_final_partida = "Final de la partida: "
        text_final_partida_2 = ""
        if len(self.__game.get_tutes()) != 0:
            # poden guanyar fins a 2 jugadors alhora
            for i, player_id in enumerate(self.__game.get_tutes()):
                player = self.__game.get_player_by_position(player_id)
                has_tute, tute_kings = player.hand_has_tute()
                tute_cards = "reis" if tute_kings else "cavalls"
                text_final_partida += "El jugador " + str(player.get_id()) + " ha guanyat per tute de " + tute_cards
                if i + 1 != len(self.__game.score_get_last_winners()):
                    text_final_partida += " -- "
                    # mostrar les cartes com quan té per cantar
                card_positions: Optional[List[int]] = player.hand_tute_cards_position()
                if card_positions is not None:
                    self.__show_cards_position(player, card_positions)

                self.__frames.player_score_update_total_score(player_id, self.__game.score_get_total_scores()[player_id], self.__game.score_get_wins()[player_id])
        if self.__game.get_black_hand() != -1:
            # cas que algu guanyi per mà negre i un altre per tute de reis
            if len(self.__game.get_tutes()) != 0:
                text_final_partida += " -- "

            # només pot guanyar 1 jugador amb mà negre
            player = self.__game.get_player_by_position(self.__game.get_black_hand())
            text_final_partida += "El jugador " + str(player.get_id()) + " ha guanyat per mà negre"
            # mostrar les cartes com quan té per cantar
            card_positions = player.hand_black_hand_cards_position(self.__game.deck_get_trump_suit_id())
            if card_positions is not None:
                self.__show_cards_position(player, card_positions)

            self.__frames.player_score_update_total_score(player.get_id(), self.__game.score_get_total_scores()[player.get_id()], self.__game.score_get_wins()[player.get_id()])
        if self.__game.get_hunt_the_three() != -1:
            player = self.__game.get_player_by_position(self.__game.get_hunt_the_three())
            # només pot guanyar 1 jugador
            text_final_partida += "El jugador " + str(player.get_id()) + " ha caçat al 3"
            self.__frames.player_score_update_total_score(player.get_id(), self.__game.score_get_total_scores()[player.get_id()], self.__game.score_get_wins()[player.get_id()])
        if not self.__game.is_tute() and not self.__game.is_black_hand() and not self.__game.is_hunt_the_three():
            text_final_partida_2 = "Guanya "

            for i in range(self.__num_players):
                # Actualitzar puntuació
                if self.__game.is_single_mode():
                    self.__frames.player_score_update_total_score(i, self.__game.score_get_total_scores()[i], self.__game.score_get_wins()[i])

                    text_final_partida += " Jugador " + str(i) + " - " + str(self.__game.score_get_history_scores()[-1][i]) + " punts"
                    if self.__num_players != i + 1:
                        text_final_partida += " --"
                # elif self.num_players < 3:
                else:
                    team_id = i % 2
                    self.__frames.player_score_update_total_score(i, self.__game.score_get_total_scores()[team_id], self.__game.score_get_wins()[team_id])

                    if i < 2:
                        text_final_partida += "equip de jugadors " + str(i) + " / " + str(i + 2) + " - " + str(self.__game.score_get_total_scores()[team_id]) + " punts"
                        if i == 0:
                            text_final_partida += " -- "

            for index, i in enumerate(self.__game.score_get_last_winners()):
                if self.__game.is_single_mode():
                    text_final_partida_2 += "jugador " + str(i)
                    if index + 1 != len(self.__game.score_get_last_winners()):
                        text_final_partida_2 += " -- "
                else:
                    text_final_partida_2 += "equip de jugadors " + str(i) + " / " + str(i + 2)
                    if index + 1 != len(self.__game.score_get_last_winners()):
                        text_final_partida_2 += " -- "

        self.__frames.info.add_message(text_final_partida)

        if not self.__game.is_tute() and not self.__game.is_black_hand() and not self.__game.is_hunt_the_three() and text_final_partida_2 != "":
            self.__frames.info.add_message(text_final_partida_2)

        self.__buttons.btn_new_game.enable_button()

    def __new_game(self) -> None:
        self.__buttons.btn_new_game.disable_button()

        self.__reset()

        self.__last_initial_player += 1

        for i in range(self.__num_players):
            self.__frames.player_score_update_score(i, 0)

        self.__game.new_game()

        self.__deck_counter = Playable_deck_counter(self.__frames.frm_board, self.__game.deck_get_deck_size() + 1)

        self.__root.after(SLEEP_TIME, self.__deal_cards, self.__game.get_num_cards(), True)  # Després de 2 segons, començar a repartir les cartes

    def __next_round(self) -> None:
        self.__game.next_round(self.__first_round)

        # Deshabilitem tots els botons de nou
        self.__buttons.disable_buttons()

        if self.__first_round:
            if not self.__game.continue_game():
                self.__end_game()
        else:
            # Eliminar imatges dels labels de les cartes jugades
            # Reiniciar cartes jugades
            self.__remove_played_cards()
            # print(self.played_cards_position)

            # Repartir cartes (si en queden)
            if self.__continue_dealing:
                self.__deal_cards(1)
                # Cal comprovar si el joc a finalitzat
                if not self.__game.continue_game():
                    self.__end_game()
            else:
                tksleep(self.__root, MOVING_TO_NEW_HAND_POS_TIME)

            # Es comprova si caldrà seguir repartint cartes a la següent ronda
            if not self.__game.deck_has_remaining_cards():
                self.__continue_dealing = False

            self.__last_initial_player = self.__game.get_last_round_winner_id()

            # TEST
            # tksleep(4)

            # Comprovacio on son les cartes....
            # for player_id in range(4):
            #   for playable_card in self.player_cards_lbl[player_id]:
            #       print(player_id, playable_card.real_pos_x, playable_card.real_pos_y)

        if not self.__game_has_ended:
            wait_until_choose_sing_suit = False if self.__game.is_brisca() else self.__show_singed_cards()

            self.__round_turn_idx = 0
            self.__cards_played = 0

            # No podem seguir jugant si hem d'esperar a que trii un cant
            if not wait_until_choose_sing_suit:
                if self.__first_round:
                    self.__first_round = False

                # Training and AI
                self.__game.game_state_set_sing_declarations()

                self.__play_card()

    def __nn_action(self, player: Player) -> None:
        # Executar torn de la màquina
        card_position, card_or_change = self.__game.player_turn(player)

        if card_or_change is None:
            self.__frames.info.add_message("El jugador " + str(player.get_id()) + " ha intercanviat la carta de triomf")

            # print("card_or_change - card_position", card_position)
            # Hem d'intertcanviar la carta de la posicio "card_position" amb la del triomf
            self.__change_card(card_position, player.get_id())

            # Training and AI
            self.__game.game_state_set_action(player.get_id(), 0, 0)
            self.__game.game_state_add_current_to_round(player.get_id())

            self.__game.game_state_change_trump_card()

            # Comprovar mà negre després d'intercanvi
            if not self.__game.is_black_hand():
                # tornar a jugar
                card_position, card_or_change = self.__game.player_turn(player)
                # print("card_or_change - card_position", card_position)

                # Pot intercanviar 2 cops en 1 tirada
                if card_or_change is None:
                    self.__frames.info.add_message("El jugador " + str(player.get_id()) + " ha intercanviat la carta de triomf")
                    # Hem d'intertcanviar la carta de la posicio "card_position" amb la del triomf
                    self.__change_card(card_position, player.get_id())

                    # Training and AI
                    self.__game.game_state_set_action(player.get_id(), 0, 0)
                    self.__game.game_state_add_current_to_round(player.get_id())

                    # Comprovar mà negre després d'intercanvi
                    if not self.__game.is_black_hand():
                        # tornar a jugar
                        card_position, card_or_change = self.__game.player_turn(player)

        # Training and AI
        # self.__game.game_state_set_action(player.get_id(), card_position + 1)
        # type -> 0=canvi, 1=carta, 2=cant
        if card_or_change is None:
            raise AssertionError("card is None")
        self.__game.game_state_set_action(player.get_id(), card_or_change.get_training_idx(), 1)
        # self.__game.game_state_add_played_card(card_or_change)

        self.__game.game_state_add_current_to_round(player.get_id())
        self.__game.game_state_new_turn(player.get_id())

        # Comprovar mà negre després d'intercanvi
        if not self.__game.is_black_hand():
            card_pos_x, card_pos_y = calc_hand_card_position(player.get_id(), card_position, self.__game.get_num_cards())

            # Hem de mostrar la carta i moure-la a la posició correcta dins de la zona de joc
            # print("movent carta del rival: ", player.get_id())
            playable_card = self.__player_cards_lbl[player.get_id()][card_position]
            playable_card.change_render(get_card_path(card_or_change), True)
            playable_card.change_place(card_pos_x, card_pos_y)

            # S'ha de moure la carta jugada al centre
            x, y = calc_played_card_position(self.__num_players, self.__cards_played)
            # playable_card.init_iterate()
            playable_card.move_to_pos(x, y)
            # playable_card.change_place(x, y)

            if not INSTANT_DEAL:
                tksleep(self.__root, DEAL_SLEEP_TIME)
            else:
                tksleep(self.__root, DEAL_INSTANT_SLEEP_TIME)

            self.__played_cards_position[player.get_id()] = card_position

            self.__cards_played += 1

        # Següent jugador
        self.__round_turn_idx += 1
        self.__play_card()

    def __play_card(self) -> None:
        # print("play_card", self.round_turn_idx)
        if self.__round_turn_idx == self.__num_players:
            # Comprovacions ronda i next round
            # print("comprovacions ronda!")
            if not self.__game.is_black_hand():
                self.__game.finalize_round()

                if self.__game.round_has_singed():
                    points = "40" if self.__game.round_is_sing_suit() else "20"
                    self.__frames.info.add_message("El jugador " + str(self.__game.get_last_round_winner_id()) + " guanya la ronda i s'emporta " + str(self.__game.get_last_round().get_round_points()) + " punts (+" + points + " en " + SUITS[self.__game.round_get_singed_suit()] + ")")
                else:
                    self.__frames.info.add_message("El jugador " + str(self.__game.get_last_round_winner_id()) + " guanya la ronda i s'emporta " + str(self.__game.get_last_round().get_round_points()) + " punts")

                # Actualitzar puntuació
                # print(self.__game.score_get_individual_scores()[self.__game.get_last_round_winner_id()])
                self.__frames.player_score_update_score(self.__game.get_last_round_winner_id(), self.__game.score_get_individual_scores()[self.__game.get_last_round_winner_id()])

            if self.__game.continue_game():
                self.__next_round()
            else:
                self.__end_game()
        else:
            # Comprovar mà negre després d'intercanvi
            if not self.__game.is_black_hand():
                player: Player = self.__game.get_player_turn(self.__round_turn_idx)
                # print("player turn: ", player.id)
                if self.__human_player and player.get_id() == 0:
                    self.__enable_cards_drag(player)

                    # Comprovar si pot canviar carta i habilitar botó
                    self.__check_can_change(player)

                    # Training and AI
                    self.__game.game_state_new_turn(player.get_id())
                else:
                    self.__nn_action(player)
            else:
                # Si hi ha mà negre, passem al següent jugador sense tirar cap carta fins arribar a final de ronda on es calcularà el guanyador
                self.__round_turn_idx += 1
                self.__play_card()

    def __player_playable_cards(self, player: Player) -> List[int]:
        highest_suit_card: Optional[Card] = None if self.__game.is_brisca() else self.__game.round_highest_suit_card()
        deck_has_cards: bool = False if self.__game.is_brisca() else self.__game.deck_has_remaining_cards()
        highest_trump_played: Optional[Card] = None if self.__game.is_brisca() else self.__game.round_highest_trump_played()

        return player.hand_get_playable_cards_positions(self.__game.deck_get_trump_suit_id(), highest_suit_card, deck_has_cards, highest_trump_played)

    def __reset(self) -> None:
        self.__game.score_reset_last_winners()

        # print(self.__player_cards_lbl)
        for i in range(self.__num_players):
            for pc_lbl in self.__player_cards_lbl[i]:
                pc_lbl.destroy_label()

            self.__frames.player_score_update_score(i, 0)

        self.__player_cards_lbl = []
        for i in range(0, self.__num_players):
            self.__player_cards_lbl.append([])

        self.__game_has_ended = False
        self.__continue_dealing = True
        self.__first_round = True

        self.__played_cards_position = [None] * self.__num_players
        self.__card_dragged_position = -1
        self.__round_turn_idx = 0

        if self.__lbl_trump_card is not None:
            self.__lbl_trump_card.destroy_label()
            self.__lbl_trump_card = None

        if self.__lbl_deck is not None:
            self.__lbl_deck.destroy_label()

        self.__lbl_deck = Playable_card(self.__root, self.__frames.frm_board, get_card_path(None), None, None, None, DECK_POS_X, DECK_POS_Y, True)

    # Buttons interaction
    def __check_can_change(self, player: Player) -> None:
        # Comprovar si pot canviar carta i habilitar botó
        if self.__game.is_rule_active('can_change') and self.__game.deck_has_remaining_cards() and player.hand_can_change(self.__game.deck_is_high_trump(), self.__game.deck_get_trump_suit_id()):
            self.__frames.info.add_message("El jugador 0 pot intercanviar la carta de triomf")
            self.__buttons.btn_change_card.enable_button()
        elif self.__game.is_rule_active('can_change'):
            self.__buttons.btn_change_card.disable_button()

    def __choose_change_card(self) -> None:
        # Deshabilitem el botó
        self.__buttons.btn_change_card.disable_button()

        player = self.__game.get_player_by_position(0)

        # Deshabilitem el drag de totes les cartes
        for playable_card in self.__player_cards_lbl[player.get_id()]:
            self.__make_not_draggable(playable_card.get_label())

        # TODO s'haura d'emmagatzemar d'alguna manera
        # Aquest codi esta repetit, mirar si el puc agrupar d'alguna manera
        # print("Acció: intercanviar carta")
        # print(self.deck.is_high_trump(), self.deck.trump_card_suit(), player.hand)

        card_position, card_to_change = player.hand_card_to_change(self.__game.deck_is_high_trump(), self.__game.deck_get_trump_suit_id())
        trump_card = self.__game.deck_get_trump_card()
        # print(card_to_change)
        # print("per")
        # print(trump_card)
        if trump_card is not None and card_to_change is not None:
            player.hand_change_card(card_position, trump_card)
            self.__game.deck_change_trump_card(card_to_change)
            self.__change_card(card_position, player.get_id())
            self.__game.game_state_change_trump_card()

            # Training and AI
            self.__game.game_state_set_action(player.get_id(), 0, 0)
            self.__game.game_state_add_current_to_round(player.get_id())
            self.__game.game_state_new_turn(0)
        else:
            raise AssertionError("Trump card or card to change is None")

        # Comprovar mà negre
        if self.__game.is_rule_active('black_hand') and player.hand_has_black_hand(self.__game.deck_get_trump_suit_id()):
            # print("té mà negre")
            self.__game.set_black_hand(player.get_id())
            # Següent jugador
            self.__round_turn_idx += 1
            self.__play_card()
        else:
            # Comprovem si pot tornar a canviar
            self.__check_can_change(player)

            # Habilitem el drag de totes les cartes que poden jugar-se
            self.__enable_cards_drag(player)

    def __choose_sing_suit(self, chosen_suit_id: int) -> None:
        # Training and AI
        self.__game.game_state_new_turn(0)
        self.__game.game_state_set_action(0, chosen_suit_id, 2)
        self.__game.game_state_add_current_to_round(0)
        self.__game.game_state_new_turn(0)
        self.__first_round = False

        self.__buttons.btn_sing_gold.disable_button()
        self.__buttons.btn_sing_coarse.disable_button()
        self.__buttons.btn_sing_cups.disable_button()
        self.__buttons.btn_sing_swords.disable_button()

        if chosen_suit_id == 1:
            self.__buttons.btn_sing_gold.clicked_button()
        elif chosen_suit_id == 2:
            self.__buttons.btn_sing_coarse.clicked_button()
        elif chosen_suit_id == 3:
            self.__buttons.btn_sing_swords.clicked_button()
        elif chosen_suit_id == 4:
            self.__buttons.btn_sing_cups.clicked_button()

        if chosen_suit_id == self.__game.deck_get_trump_suit_id():
            self.__frames.info.add_message("El jugador 0 té les 40")
        else:
            self.__frames.info.add_message("El jugador 0 té les 20 en " + SUITS[chosen_suit_id])

        # Indiquem el cant i seguim jugant
        sing_dec: Optional[List[Optional[int]]] = self.__game.round_get_sing_declarations()
        if sing_dec is not None:
            sing_dec[0] = chosen_suit_id

        self.__show_and_hide_singed(self.__game.get_player_by_position(0), chosen_suit_id)

        # Training and AI
        self.__game.game_state_set_sing_declarations()

        self.__play_card()

    # Card iterations
    def __change_card(self, card_pos: int, player_id: int) -> None:
        if self.__lbl_trump_card is not None:
            # La carta de la mà del jugador ja és la que hi havia de triomf
            # Per tant, la carta de triomf del deck és la que tenia el jugador a la mà
            card_pos, trump_card = self.__game.get_player_by_position(player_id).hand_get_card_in_position_no_remove(card_pos)

            card = self.__game.deck_get_trump_card()

            hide = not SHOW_RIVAL_CARDS
            playable_card = self.__player_cards_lbl[player_id][card_pos]

            vertical_position = player_id % 2 == 0

            path_img = get_card_path(card)

            if self.__human_player and player_id == 0:
                hide = False

            # La carta no es veu, primer s'ha de fer el canvi de la imatge i fer els intercanvis
            playable_card.change_render(path_img, False)

            # S'ha de fer l'animació de les dues cartes i després tornar a fer l'intercanvi instantani de les imatges dels labels
            # (com que els tinc dins dels array a les posicions, aquests labels sempre han de tornar al mateix lloc)

            # girar (si cal) i moure lbl a la posició del triomf
            playable_card.move_to_pos(TRUMP_CARD_POS_X, TRUMP_CARD_POS_Y)

            if not INSTANT_DEAL:
                tksleep(self.__root, CHANGE_SLEEP_TIME)
            else:
                tksleep(self.__root, DEAL_INSTANT_SLEEP_TIME)

            # girar (si cal) i moure lbl_trump a la mà
            # Ara s'ha de girar normal

            self.__lbl_trump_card.change_render(get_card_path(trump_card), vertical_position)
            self.__lbl_trump_card.move_to_pos(playable_card.get_hand_pos_x_not_none(), playable_card.get_hand_pos_y_not_none())

            if not INSTANT_DEAL:
                tksleep(self.__root, CHANGE_SLEEP_TIME)
            else:
                tksleep(self.__root, DEAL_INSTANT_SLEEP_TIME)

            # Fi dels intercanvis, ara es posa tot a lloc de nou
            if hide:
                playable_card.change_render(get_card_path(None), vertical_position)
                playable_card.change_place(playable_card.get_hand_pos_x_not_none(), playable_card.get_hand_pos_y_not_none())
            else:
                playable_card.change_render(get_card_path(trump_card), vertical_position)
                playable_card.change_place(playable_card.get_hand_pos_x_not_none(), playable_card.get_hand_pos_y_not_none())

            self.__lbl_trump_card.change_render(path_img, False)
            self.__lbl_trump_card.change_place(TRUMP_CARD_POS_X, TRUMP_CARD_POS_Y)

    def __deal_card(self, card: Card, player_id: int, card_pos: int, is_trump_card: bool) -> None:
        # El primer cop que es creen les cartes s'haurà de calcular la posició a pantalla
        # La resta estarà ja calculat a la classe
        vertical_position = player_id % 2 == 0

        # TODO Fem que la carta de triomf quedi amb transparència
        # No es pot fer així, utilitzar canvas i label (buscar-ho)
        # if is_trump_card:
            # self.lbl_trump_card.lbl.attribute('-alpha', 0.3)

        card_pos_x, card_pos_y = calc_hand_card_position(player_id, card_pos, self.__game.get_num_cards())

        if self.__human_player and player_id == 0:
            path_to_img = get_card_path(card)

            playable_card = Playable_card(self.__root, self.__frames.frm_board, path_to_img, card_pos, card_pos_x, card_pos_y, DECK_POS_X, DECK_POS_Y, vertical_position)
            self.__player_cards_lbl[player_id].append(playable_card)

            playable_card.move_to_hand()
        else:
            # No s'ha de veure la carta (a no ser que sigui la carta de triomf, que amagarem al final)
            if is_trump_card:
                path_to_img = get_card_path(card)
                if SHOW_RIVAL_CARDS:
                    path_to_img_trump = path_to_img
                else:
                    path_to_img_trump = get_card_path(None)
            else:
                if SHOW_RIVAL_CARDS:
                    path_to_img = get_card_path(card)
                else:
                    path_to_img = get_card_path(None)

            playable_card = Playable_card(self.__root, self.__frames.frm_board, path_to_img, card_pos, card_pos_x, card_pos_y, DECK_POS_X, DECK_POS_Y, vertical_position)
            self.__player_cards_lbl[player_id].append(playable_card)

            playable_card.move_to_hand()
        if not INSTANT_DEAL:
            tksleep(self.__root, DEAL_SLEEP_TIME)

        if is_trump_card and player_id != 0:
            playable_card.change_render(path_to_img_trump, vertical_position)

    def __remove_played_cards(self) -> None:
        if not INSTANT_DEAL:
            # TODO Optimitzar codi repetit
            for i in range(self.__num_players):
                playable_card_pos = self.__played_cards_position[i]

                if playable_card_pos is None:
                    raise AssertionError("Playable card pos is None")

                playable_card = self.__player_cards_lbl[i][playable_card_pos]

                # S'elimina el label d'aquesta carta
                playable_card.destroy_label()

                # S'elimina la carta de la llista de cartes del jugador
                self.__player_cards_lbl[i].remove(playable_card)

                # Ara, cal moure totes les cartes a la dreta d'aquesta posició una posició a l'esquerra
                for j in range(playable_card_pos, len(self.__player_cards_lbl[i])):
                    card_pos_x, card_pos_y = calc_hand_card_position(i, j, self.__game.get_num_cards())
                    lbl = self.__player_cards_lbl[i][j]
                    lbl.change_hand_position(card_pos_x, card_pos_y)
        else:
            for i in range(self.__num_players):
                playable_card_pos = self.__played_cards_position[i]

                if playable_card_pos is None:
                    raise AssertionError("Playable card pos is None")

                playable_card = self.__player_cards_lbl[i][playable_card_pos]

                # S'elimina el label d'aquesta carta
                playable_card.destroy_label()

                # S'elimina la carta de la llista de cartes del jugador
                self.__player_cards_lbl[i].remove(playable_card)

            for i in range(self.__num_players):
                playable_card_pos = self.__played_cards_position[i]

                if playable_card_pos is None:
                    raise AssertionError("Playable card pos is None")

                # Ara, cal moure totes les cartes a la dreta d'aquesta posició una posició a l'esquerra
                for j in range(playable_card_pos, len(self.__player_cards_lbl[i])):
                    card_pos_x, card_pos_y = calc_hand_card_position(i, j, self.__game.get_num_cards())
                    lbl = self.__player_cards_lbl[i][j]
                    lbl.change_hand_position(card_pos_x, card_pos_y)

    def __show_cards_position(self, player: Player, cards_position: List[int]) -> None:
        # Hem de mostrar les cartes
        for card_pos in cards_position:
            card_pos, card = player.hand_get_card_in_position_no_remove(card_pos)
            path_to_img = get_card_path(card)
            vertical_position = player.get_id() % 2 == 0

            playable_card: Playable_card = self.__player_cards_lbl[player.get_id()][card_pos]
            playable_card.show_important_card(path_to_img, vertical_position, player.get_id())

        tksleep(self.__root, SING_TIME)

    def __show_and_hide_singed(self, player: Player, sing_suit: int) -> None:
        # Hem de mostrar les cartes
        # Busquem les posicions de les cartes del cant i les mostrem
        card_positions: List[int] = player.hand_get_sing_cards_position(sing_suit)
        for card_pos in card_positions:
            card_pos, card = player.hand_get_card_in_position_no_remove(card_pos)
            path_to_img = get_card_path(card)
            vertical_position = player.get_id() % 2 == 0

            playable_card: Playable_card = self.__player_cards_lbl[player.get_id()][card_pos]
            playable_card.show_important_card(path_to_img, vertical_position, player.get_id())

        tksleep(self.__root, SING_TIME)

        # Ara les tornem a amagar
        for card_pos in card_positions:
            card_pos, card = player.hand_get_card_in_position_no_remove(card_pos)

            path_to_img = get_card_path(card)
            if player.get_id() != 0 and not SHOW_RIVAL_CARDS:
                path_to_img = get_card_path(None)

            vertical_position = player.get_id() % 2 == 0

            playable_card = self.__player_cards_lbl[player.get_id()][card_pos]
            playable_card.hide_important_card(path_to_img, vertical_position, player.get_id())

        tksleep(self.__root, AFTER_SING_TIME)

    def __show_singed_cards(self) -> bool:
        wait_until_choose_sing_suit = False

        # Si hi ha algun tute, mostrarem les cartes i les diferenciarem una mica canviant la posició
        sing_dec: Optional[List[Optional[int]]] = self.__game.round_get_sing_declarations()
        if sing_dec is not None:
            for i, sing_declaration in enumerate(sing_dec):
                player = self.__game.get_player_by_position(i)
                # print("sing_declaration", sing_declaration)
                if i == 0 and sing_declaration is None and self.__human_player:
                    # Comprovem si en té 2 per mostrar-li els botons
                    tute_declarations: List[int] = player.hand_sing_suits_in_hand()
                    if len(tute_declarations) > 1:
                        # Habilitar els botons corresponents
                        wait_until_choose_sing_suit = True
                        for td in tute_declarations:
                            if td == 1:
                                self.__buttons.btn_sing_gold.enable_button()
                            elif td == 2:
                                self.__buttons.btn_sing_coarse.enable_button()
                            elif td == 3:
                                self.__buttons.btn_sing_swords.enable_button()
                            elif td == 4:
                                self.__buttons.btn_sing_cups.enable_button()
                elif sing_declaration is not None:
                    if self.__first_round:
                        self.__game.game_state_new_turn(player.get_id())
                        self.__game.game_state_set_action(player.get_id(), sing_declaration, 2)
                        self.__game.game_state_add_current_to_round(player.get_id())
                        self.__game.game_state_new_turn(player.get_id())
                    else:
                        self.__game.game_state_set_action(player.get_id(), sing_declaration, 2)
                        self.__game.game_state_add_current_to_round(player.get_id())
                        self.__game.game_state_new_turn(player.get_id())


                    if sing_declaration == self.__game.deck_get_trump_suit_id():
                        self.__frames.info.add_message("El jugador " + str(i) + " té les 40")
                    else:
                        self.__frames.info.add_message("El jugador " + str(i) + " té les 20 en " + SUITS[sing_declaration])

                    self.__show_and_hide_singed(player, sing_declaration)

                    if i == 0 and self.__human_player:
                        # El botó quedarà marcat per defecte
                        if sing_declaration == 1:
                            self.__buttons.btn_sing_gold.clicked_button()
                        elif sing_declaration == 2:
                            self.__buttons.btn_sing_coarse.clicked_button()
                        elif sing_declaration == 3:
                            self.__buttons.btn_sing_swords.clicked_button()
                        elif sing_declaration == 4:
                            self.__buttons.btn_sing_cups.clicked_button()

        return wait_until_choose_sing_suit

    # Draggable functions
    def __enable_cards_drag(self, player: Player) -> None:
        playable_positions = self.__player_playable_cards(player)

        for playable_position in playable_positions:
            playable_card = self.__player_cards_lbl[player.get_id()][playable_position]
            self.__make_draggable(playable_card.get_label())
            playable_card.add_playable_border()

    def __make_draggable(self, widget: Label) -> None:
        widget.bind("<Button-1>", self.__on_drag_start)
        widget.bind("<B1-Motion>", self.__on_drag_motion)
        widget.bind("<ButtonRelease>", self.__on_drag_stop)
        raise_frame_or_label(widget)

    def __make_not_draggable(self, widget: Label) -> None:
        widget.unbind("<Button-1>")
        widget.unbind("<B1-Motion>")
        widget.unbind("<ButtonRelease>")

    def __on_drag_start(self, event: Event) -> None:
        widget = event.widget
        widget._drag_start_x = event.x
        widget._drag_start_y = event.y
        widget._drag_start_x_pos = widget.winfo_x()
        widget._drag_start_y_pos = widget.winfo_y()

        self.__card_dragged_position = calc_dragged_card_position(widget._drag_start_x_pos, True, self.__game.get_num_cards())

    def __on_drag_motion(self, event: Event) -> None:
        widget = event.widget
        if hasattr(widget, '_drag_start_x'):
            x = widget.winfo_x() - widget._drag_start_x + event.x
            y = widget.winfo_y() - widget._drag_start_y + event.y
            widget.place(x=x, y=y)

    def __on_drag_stop(self, event: Event) -> None:
        widget = event.widget
        if hasattr(widget, '_drag_start_x'):
            x = widget.winfo_x() - widget._drag_start_x + event.x
            y = widget.winfo_y() - widget._drag_start_y + event.y

            played = False

            if not (PLAYABLE_FRAME_POS_X < x < PLAYABLE_FRAME_POS_X + PLAYABLE_FRAME_SIZE_X and PLAYABLE_FRAME_POS_Y < y < PLAYABLE_FRAME_POS_Y + PLAYABLE_FRAME_SIZE_Y) or self.__num_players == self.__cards_played:
                x = widget._drag_start_x_pos
                y = widget._drag_start_y_pos
            else:
                x, y = calc_played_card_position(self.__num_players, self.__cards_played)

                self.__make_not_draggable(widget)

                self.__cards_played += 1

                self.__player_playable_cards_undrag_all()

                # Deshabilitar botó canvi
                if self.__game.is_rule_active('can_change'):
                    self.__buttons.btn_change_card.disable_button()

                # Informar de la jugada
                player = self.__game.get_player_turn(self.__round_turn_idx)
                p, card = player.hand_get_card_in_position(self.__card_dragged_position)

                # Faig còpia de les cartes jugables abans de jugar la carta
                copied_playable_hand = copy.deepcopy(self.__game.player_hand_get_playable_cards(player.get_id()))

                self.__game.round_played_card(card)

                # Training and AI
                # type -> 0=canvi, 1=carta, 2=cant
                self.__game.game_state_set_action(player.get_id(), card.get_training_idx(), 1)
                self.__game.game_state_add_played_card(card)
                self.__game.game_state_remove_viewed_card(player.get_id(), card)
                self.__game.game_state_heuristics(player.get_id(), card, copied_playable_hand)
                self.__game.game_state_add_current_to_round(player.get_id())
                self.__game.game_state_new_turn(player.get_id())

                self.__played_cards_position[player.get_id()] = self.__card_dragged_position

                self.__card_dragged_position = -1
                self.__round_turn_idx += 1
                played = True

            widget.place(x=x, y=y)

            if played:
                tksleep(self.__root, PLAYER_PLAY_CARD_SLEEP_TIME)
                self.__play_card()

    def __player_playable_cards_undrag_all(self) -> None:
        for playable_card in self.__player_cards_lbl[0]:
            self.__make_not_draggable(playable_card.get_label())
            playable_card.remove_playable_border()
