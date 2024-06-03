from typing import List, Optional, Dict

import gym
from gym import spaces
import numpy as np

from game_environment.game import Non_playable_game, Playable_game


class Game_env(gym.Env):
    metadata = {"render_modes": ["gui"], "render_fps": 4}

    def __init__(self, game_type: int, model_type: List[int], model_path: List[Optional[str]], num_players: int, single_mode: bool, rules: Dict, render_mode=None, size=5):
        total_games: int = 1

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if self.render_mode is None:
            self.game: Non_playable_game = Non_playable_game(game_type, total_games, model_type, model_path, num_players, single_mode, rules, False, None)
        else:
            self.game: Playable_game = Non_playable_game(game_type, False, total_games, model_type, model_path, num_players, single_mode, rules, False, None)

        # 41 accions per la Brisca i 45 pel Tute
        self.action_space = spaces.Discrete(41) if self.game_type == 1 else spaces.Discrete(45)

    def _get_obs(self):
        # return {"agent": self._agent_location, "target": self._target_location}
        return {}

    def _get_info(self):
        # return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # iniciar partida i retornar observacio i info

        observation = self._get_obs()
        info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, info

    def step(self, action):

        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        pass

    def close(self):
        pass