# Boilerplate for two-phase environment.

# In the case of Balatro, the executor agent plays a single round of the game, and the modifier agent
# decides what improvements to purchase from the shop, modifying the game state.

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from balatro_gym.balatro_game import BalatroGame

# TODO: test

class TwoPhaseEnv(gym.Env):
    def __init__(self, game: BalatroGame, action_dim=5, max_weight=10):
        '''
        action_dim: Set of all possible Jokers, cards, and packs that can be purchased from the shop.
        For any given round, the action space is masked such that only the actions available to the current round are visible.
        max_weight: Corresponds to the total money available to the modifier agent.

        An assumption is made where each item in the action space has a consistent cost.
        '''
        super(TwoPhaseEnv, self).__init__()

        self.game = game

        # Phase 1: Executor agent
        # TODO: Integrate code from algorithms folder
        
        # Phase 2: Modifier agent
        # Binary vector: select/don't select each modification
        self.modifier_action_space = spaces.MultiBinary(action_dim)
        self.action_dim = action_dim
        self.max_weight = max_weight
        self.action_weights = np.arange(1, action_dim + 1)

        self.reset()

    def reset(self):
        self.modifications = np.zeros(self.action_dim)
        self.modified_state = np.zeros(self.action_dim)
        self.executor_score = 0
        return self.modified_state

    def step(self, modifier_action, policy_gradient=True):
        # Assume that the first round is easily passed with a greedy approach (Reasonable from empirical observation)
        # That way, we assume that the modifier always goes first.

        selected = (modifier_action > 0.5).astype(int) if policy_gradient else np.array(modifier_action)
        total_weight = np.sum(selected * self.action_weights)
        if total_weight > self.max_weight:
            selected = self._project_to_budget(selected)

        self.modifications = selected
        self.modified_state = selected.astype(np.float32)

        # Run executor agent (black-box scoring function)
        self.executor_score = self.executor_policy(self.modified_state)

        # Modifier reward: executor score - cost
        cost = np.sum(self.modifications * self.action_weights)
        reward_modifier = self.executor_score - 0.1 * cost

        done = True  # One step per episode
        info = {"executor_score": self.executor_score, "cost": cost}

        return self.modified_state, reward_modifier, done, info

    def executor_policy(self, modified_state):
        # TODO: Integrate code from algorithms folder
        pass

    def _project_to_budget(self, selection):
        # Greedy projection onto weight budget
        idx_sorted = np.argsort(self.action_weights)
        total = 0
        result = np.zeros_like(selection)
        for i in idx_sorted:
            w = self.action_weights[i]
            if total + w <= self.max_weight:
                result[i] = 1
                total += w
            else:
                break
        return result
    
    def get_action_mask(self):
        mask = np.ones(self.action_dim, dtype=np.bool_)
        for i in range(self.action_dim):
            if self.action_weights[i] > self.current_budget:
                mask[i] = 0
        return mask

