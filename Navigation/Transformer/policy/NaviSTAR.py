import torch
import torch.nn as nn
from torch.nn.functional import softmax
import logging
from Environment.envs.utils.action import ActionRot, ActionXY
from Transformer.models.ST_Transformer_new import STTransformer


class ValueNetwork(nn.Module):
    def __init__(self, self_state_dim, input_dim, output_dim, args):
        super().__init__()
        self.all_past_states = []
        self.st_transformer = STTransformer(self.args, torch.device('cpu'))
        self.self_state_dim = self_state_dim
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, state, past_states):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)
        probability = np.random.random()
            
            
        size = state.shape
        pred_state = self.st_transformer(state, past_states)
        value = self.pred_state
        

        return value

