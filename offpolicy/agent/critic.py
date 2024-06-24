import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import offpolicy.utils as utils
from models.STAR import STAR


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, config, obs_dim, action_dim):
        super().__init__()
        # if isinstance(obs_dim, dict):
        #     # obs_dim = obs_dim['obs_for_sac_critic'].shape[0]

        self.star = STAR(obs_dim, config)
        obs_dim = self.star.output_size
        self.Q1 = utils.mlp(obs_dim + action_dim, config.sac.hidden_dim, 1, config.sac.hidden_depth)
        self.Q2 = utils.mlp(obs_dim + action_dim, config.sac.hidden_dim, 1, config.sac.hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        # if isinstance(obs, dict):
        #     obs = obs['obs_for_sac_critic']
        _, features, _ = self.star(obs, infer=True)
        obs_action = torch.cat([features, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)
