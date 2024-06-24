import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np


def _flatten_helper(T, N, _tensor):
    if isinstance(_tensor, dict):
        for key in _tensor:
            _tensor[key] = _tensor[key].reshape(T * N, *(_tensor[key].shape[2:]))
        return _tensor
    else:
        return _tensor.reshape(T * N, *_tensor.shape[2:])

class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 human_node_rnn_size, human_human_edge_rnn_size, recurrent_cell_type):

        if isinstance(obs_shape, dict):
            self.obs = {}
            for key in obs_shape:
                self.obs[key] = np.zeros((num_steps + 1, num_processes, *(obs_shape[key].shape)))
            self.human_num = obs_shape['spatial_edges'].shape[0]
        else:
            self.obs = np.zeros((num_steps + 1, num_processes, *obs_shape))

        double_rnn_size = 1 if recurrent_cell_type=="GRU" else 2

        self.recurrent_hidden_states = {} # a dict of tuple(hidden state, cell state)

        node_num = 1
        edge_num = self.human_num + 1

        self.recurrent_hidden_states['human_node_rnn'] = np.zeros((num_steps + 1, num_processes, node_num, human_node_rnn_size*double_rnn_size))
        self.recurrent_hidden_states['human_human_edge_rnn'] = np.zeros((num_steps + 1, num_processes, edge_num, human_human_edge_rnn_size*double_rnn_size))

        self.rewards = np.zeros((num_steps, num_processes, 1))
        self.value_preds = np.zeros((num_steps + 1, num_processes, 1))
        self.returns = np.zeros((num_steps + 1, num_processes, 1))
        self.action_log_probs = np.zeros((num_steps, num_processes, 1))
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = np.zeros((num_steps, num_processes, action_shape))
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = np.ones((num_steps + 1, num_processes, 1))

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = np.ones((num_steps + 1, num_processes, 1))

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        for key in self.obs:
            self.obs[key] = self.obs[key].to(device)
        for key in self.recurrent_hidden_states:
            self.recurrent_hidden_states[key] = self.recurrent_hidden_states[key].to(device)

        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks):


        for key in self.obs:
            self.obs[key][self.step + 1] = obs[key].copy()
        for key in recurrent_hidden_states:
            self.recurrent_hidden_states[key][self.step + 1] = recurrent_hidden_states[key].copy()

        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        self.bad_masks[self.step + 1] = bad_masks.copy()

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):

        for key in self.obs:
            self.obs[key][0] = self.obs[key][-1].copy()
        for key in self.recurrent_hidden_states:
            self.recurrent_hidden_states[key][0] = self.recurrent_hidden_states[key][-1].copy()

        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True):
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = (self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape(0))):
                    self.returns[step] = self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.shape[:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:


            obs_batch = {}
            for key in self.obs:
                obs_batch[key] = self.obs[key][:-1].reshape(-1, *self.obs[key].shape[2:])[indices]
            recurrent_hidden_states_batch = {}
            for key in self.recurrent_hidden_states:
                recurrent_hidden_states_batch[key] = self.recurrent_hidden_states[key][:-1].reshape(
                -1, self.recurrent_hidden_states[key].shape[-1])[indices]

            actions_batch = self.actions.reshape(-1,
                                              self.actions.shape[-1])[indices]
            value_preds_batch = self.value_preds[:-1].reshape(-1, 1)[indices]
            return_batch = self.returns[:-1].reshape(-1, 1)[indices]
            masks_batch = self.masks[:-1].reshape(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.reshape(-1,
                                                                    1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.reshape(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.shape[1]
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):

            obs_batch = {}
            for key in self.obs:
                obs_batch[key] = []
            recurrent_hidden_states_batch = {}
            for key in self.recurrent_hidden_states:
                recurrent_hidden_states_batch[key] = []

            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]


                for key in self.obs:
                    obs_batch[key].append(self.obs[key][:-1, ind])
                for key in self.recurrent_hidden_states:
                    recurrent_hidden_states_batch[key].append(self.recurrent_hidden_states[key][0:1, ind])

                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)

            actions_batch = np.stack(actions_batch, 1)
            value_preds_batch = np.stack(value_preds_batch, 1)
            return_batch = np.stack(return_batch, 1)
            masks_batch = np.stack(masks_batch, 1)
            old_action_log_probs_batch = np.stack(
                old_action_log_probs_batch, 1)
            adv_targ = np.stack(adv_targ, 1)

            for key in obs_batch:
                obs_batch[key] = np.stack(obs_batch[key], 1)
            for key in recurrent_hidden_states_batch:
                temp = np.stack(recurrent_hidden_states_batch[key], 1)
                recurrent_hidden_states_batch[key] = temp.reshape(N, *(temp.shape[2:]))

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
