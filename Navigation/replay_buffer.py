import csv

import numpy as np
import torch
import utils
import pandas as pd

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device, window=1):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.float64
        # obs_dtype = np.float64

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        self.window = window

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
    
    def add_batch(self, obs, action, reward, next_obs, done, done_no_max):
        
        next_index = self.idx + self.window
        if next_index >= self.capacity:
            self.full = True
            maximum_index = self.capacity - self.idx
            np.copyto(self.obses[self.idx:self.capacity], obs[:maximum_index])
            np.copyto(self.actions[self.idx:self.capacity], action[:maximum_index])
            np.copyto(self.rewards[self.idx:self.capacity], reward[:maximum_index])
            np.copyto(self.next_obses[self.idx:self.capacity], next_obs[:maximum_index])
            np.copyto(self.not_dones[self.idx:self.capacity], done[:maximum_index] <= 0)
            np.copyto(self.not_dones_no_max[self.idx:self.capacity], done_no_max[:maximum_index] <= 0)
            remain = self.window - (maximum_index)
            if remain > 0:
                np.copyto(self.obses[0:remain], obs[maximum_index:])
                np.copyto(self.actions[0:remain], action[maximum_index:])
                np.copyto(self.rewards[0:remain], reward[maximum_index:])
                np.copyto(self.next_obses[0:remain], next_obs[maximum_index:])
                np.copyto(self.not_dones[0:remain], done[maximum_index:] <= 0)
                np.copyto(self.not_dones_no_max[0:remain], done_no_max[maximum_index:] <= 0)
            self.idx = remain
        else:
            np.copyto(self.obses[self.idx:next_index], obs)
            np.copyto(self.actions[self.idx:next_index], action)
            np.copyto(self.rewards[self.idx:next_index], reward)
            np.copyto(self.next_obses[self.idx:next_index], next_obs)
            np.copyto(self.not_dones[self.idx:next_index], done <= 0)
            np.copyto(self.not_dones_no_max[self.idx:next_index], done_no_max <= 0)
            self.idx = next_index
        
    def relabel_with_predictor(self, predictor):
        batch_size = 200
        self.idx = len(self.obses)
        total_iter = int(self.idx/batch_size)
        
        if self.idx > batch_size*total_iter:
            total_iter += 1
            
        for index in range(total_iter):
            last_index = (index+1)*batch_size
            if (index+1)*batch_size > self.idx:
                last_index = self.idx
                
            obses = self.obses[index*batch_size:last_index]
            actions = self.actions[index*batch_size:last_index]
            inputs = np.concatenate([obses, actions], axis=-1)
            
            pred_reward = predictor.r_hat_batch(inputs)
            self.rewards[index*batch_size:last_index] = pred_reward
            
    def sample(self, batch_size):

        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max
    
    def sample_state_ent(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)
        
        if self.full:
            full_obs = self.obses
        else:
            full_obs = self.obses[: self.idx]
        full_obs = torch.as_tensor(full_obs, device=self.device).float()
        return obses, full_obs, actions, rewards, next_obses, not_dones, not_dones_no_max

    def resume_train_set(self,obs, action, reward, next_obs, not_dones, not_dones_no_max):
        len_data = len(obs)
        np.copyto(self.obses[:len_data], obs[:])
        np.copyto(self.actions[:len_data], action[:])
        np.copyto(self.rewards[:len_data], reward[:])
        np.copyto(self.next_obses[:len_data], next_obs[:])
        np.copyto(self.not_dones[:len_data], not_dones[:])
        np.copyto(self.not_dones_no_max[:len_data], not_dones_no_max[:])
        self.idx = len_data % self.capacity
        self.full = self.full or self.idx == 0

    def save(self,save_dir):
        obses_path = save_dir + "obses.csv"
        actions_path = save_dir + "actions.csv"
        rewards_path = save_dir + "rewards.csv"
        next_obses_path = save_dir + "next_obses.csv"
        not_dones_path = save_dir + "not_dones.csv"
        not_dones_no_max_path = save_dir + "not_dones_no_max.csv"
        path_list = [obses_path,actions_path,rewards_path,next_obses_path,not_dones_path,not_dones_no_max_path]
        for index,item in enumerate([self.obses,self.actions,self.rewards,self.next_obses,self.not_dones,self.not_dones_no_max]):
            with open(path_list[index],"w") as f:
                writer = csv.writer(f)
                for row in item:
                    writer.writerow(row)

    def load(self,load_dir):
        data_list = [[] for i in range(6)]
        obses_path = load_dir + "obses.csv"
        actions_path = load_dir + "actions.csv"
        rewards_path = load_dir + "rewards.csv"
        next_obses_path = load_dir + "next_obses.csv"
        not_dones_path = load_dir + "not_dones.csv"
        not_dones_no_max_path = load_dir + "not_dones_no_max.csv"
        path_list = [obses_path, actions_path, rewards_path, next_obses_path, not_dones_path, not_dones_no_max_path]
        for index,path in enumerate(path_list):
            with open(path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    data_list[index].append(row)

        def to_array(list):
            return np.array(list,dtype=float)
        for index,item in enumerate(data_list):
            data_list[index] = to_array(item)
        [obs, action, reward, next_obs, not_dones, not_dones_no_max] = data_list
        self.resume_train_set(obs, action, reward, next_obs, not_dones, not_dones_no_max)

    def save_reward(self,save_dir):
        rewards_path = save_dir + "rewards.csv"
        with open(rewards_path,"w") as f:
            writer = csv.writer(f)
            for row in self.rewards:
                writer.writerow(row)

