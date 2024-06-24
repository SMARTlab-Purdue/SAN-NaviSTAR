import numpy
import numpy as np
import torch
from torch import nn
from torch import distributions as pyd
import torch.nn.functional as F
import gym
import os
from collections import deque
import random
import math



def make_env(config):
    """Helper function to create dm_control environment"""
    env = gym.make(config.env.env_name)
    env.configure(config)
    envSeed = config.env.seed if config.env.seed is not None else None
    env.thisSeed = envSeed
    env.nenv = 1
    env.phase = 'train'
    return env

def make_eval_env(config):
    """Helper function to create dm_control environment"""
    env = gym.make(config.env.env_name)
    env.configure(config)
    envSeed = config.env.seed if config.env.seed is not None else None
    env.thisSeed = envSeed
    env.nenv = 1
    env.phase = 'test'
    return env

def clip_action(action, low=None, high=None, clip_norm=False, max_norm=None):
    # action [batch, action_dim] or [action_dim]
    if type(action) == numpy.ndarray:
        numpy_flag = True
        action = torch.from_numpy(action)
    elif type(action) == torch.Tensor:
        numpy_flag = False
    if clip_norm:
        action_norm = torch.norm(action, dim=-1, keepdim=True)
        action_norm = action_norm.clamp(min=max_norm)
        action = action / action_norm * max_norm

    else:
        action = action.clamp(min=low, max=high)


    if numpy_flag:
        action = action.numpy()

    return action




class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk



def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()
