from gymnasium import spaces
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ..models.market_model import calculate_market_share
from ..config import STARTING_PRICES, LARGEST_DISCOUNT, LARGEST_INCREASE, PERIODS

class InertiaEnv(MultiAgentEnv):
    def __init__(self, seed=None):
        super(InertiaEnv, self).__init__()
        self.t_steps = 0
        self.num_agents = len(STARTING_PRICES)
        self._agent_ids = [f'agent_{i}' for i in range(self.num_agents)]

        self.action_space = spaces.Dict({
            agent: spaces.Box(low=0-LARGEST_DISCOUNT, high=LARGEST_INCREASE, dtype=np.int32)
            for agent in self._agent_ids
        })

        self.observation_space = spaces.Dict({
            agent: spaces.Dict({
                'price': spaces.Box(low=0, high=max(STARTING_PRICES)+30*LARGEST_INCREASE, dtype=np.int32),
                'market_prices': spaces.Box(low=0, high=max(STARTING_PRICES)+30*LARGEST_INCREASE, shape=(len(STARTING_PRICES),), dtype=np.int32),
                'market_quantities': spaces.Box(low=0, high=max(STARTING_PRICES)+30*LARGEST_INCREASE, shape=(len(STARTING_PRICES),), dtype=np.int32),
            })
            for agent in self._agent_ids
        })
        self.reset()

    def step(self, actions):
        self.t_steps += 1
        obs = {}
        rewards = {}
        terminateds = {}
        truncateds = {}
        info = {}
        prices = self.prices

        for i, (agent_id, action) in enumerate(actions.items()):
            truncateds[agent_id] = False
            terminateds[agent_id] = False
            prices[i] += action
            prices[i] = max(0, min(prices[i], max(STARTING_PRICES) + 30 * LARGEST_INCREASE))

        self.quantities = calculate_market_share(prices)
        self.prices = prices

        for i, (agent_id, action) in enumerate(actions.items()):
            rewards[agent_id] = self.quantities[i] * prices[i]

        truncateds['__all__'] = all(truncateds.values())

        if self.current_period > PERIODS:
            for agent_id in actions.keys():
                terminateds[agent_id] = True
            terminateds['__all__'] = all(terminateds.values())

        for i, (agent_id, action) in enumerate(actions.items()):
            obs[agent_id] = self._get_obs(i)
        terminateds['__all__'] = all(terminateds.values())

        return obs, rewards, terminateds, truncateds, info

    def reset(self, *, seed=None, options=None):
        self.current_period = 0
        self.prices = STARTING_PRICES.copy()
        self.quantities = calculate_market_share(STARTING_PRICES)
        self.states = {
            agent_id: {
                'price': STARTING_PRICES[i],
                'market_prices': np.array(STARTING_PRICES),
                'market_quantities': np.array(calculate_market_share(STARTING_PRICES))
            }
            for i, agent_id in enumerate(self._agent_ids)
        }
        obs = {agent_id: self._get_obs(i) for i, agent_id in enumerate(self._agent_ids)}
        return obs, {}

    def _get_obs(self, agent_id):
        return {
            'price': self.prices[agent_id],
            'market_prices': self.prices,
            'market_quantities': self.quantities
        }