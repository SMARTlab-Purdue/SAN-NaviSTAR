import torch
import torch.nn as nn
from models.distributions import Bernoulli, Categorical, DiagGaussian
from models.srnn_model import SRNN
from models.STAR import STAR
from src.utils import check


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None, device='cpu'):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if base == 'srnn':
            base = SRNN
            self.base = base(obs_shape, base_kwargs)
            self.srnn = True
        elif base == 'star':
            base = STAR
            self.base = base(obs_shape, base_kwargs)
            self.star = True
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]

            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

        self.tqdv = dict(dtype=torch.float32, device=device)

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        for key in inputs.keys():
            inputs[key] = check(inputs[key]).to(**self.tqdv)
        for key in rnn_hxs.keys():
            rnn_hxs[key] = check(rnn_hxs[key]).to(**self.tqdv)

        masks = check(masks).to(**self.tqdv)

        if not hasattr(self, 'srnn'):
            self.srnn = False
        if not hasattr(self, 'star'):
            self.star = False

        if self.srnn:
            value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, infer=True)
        elif self.star:
            value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, infer=True)
        else:
            value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()


        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        for key in inputs.keys():
            inputs[key] = check(inputs[key]).to(**self.tqdv)
        for key in rnn_hxs.keys():
            rnn_hxs[key] = check(rnn_hxs[key]).to(**self.tqdv)

        masks = check(masks).to(**self.tqdv)
        value, _, _ = self.base(inputs, rnn_hxs, masks, infer=True)

        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        for key in inputs.keys():
            inputs[key] = check(inputs[key]).to(**self.tqdv)
        for key in rnn_hxs.keys():
            rnn_hxs[key] = check(rnn_hxs[key]).to(**self.tqdv)

        masks = check(masks).to(**self.tqdv)
        action = check(action).to(**self.tqdv)
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)

        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs



