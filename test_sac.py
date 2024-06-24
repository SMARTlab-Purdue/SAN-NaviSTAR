import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from offpolicy.replay_buffer import ReplayBuffer
import offpolicy.utils as utils
import crowd_sim
import os
from offpolicy.agent.sac import SACAgent
from crowd_sim.envs.utils.info import *

class Workspace(object):
    def __init__(self):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')


        self.model_dir = 'data/navigation/star_sac'
        self.test_dir = 'test'
        self.model = 'sac_actor593885.pt'

        from importlib import import_module
        model_dir_temp = self.model_dir
        if model_dir_temp.endswith('/'):
            model_dir_temp = model_dir_temp[:-1]
        try:
            model_dir_string = model_dir_temp.replace('/', '.') + '.configs.config'
            model_arguments = import_module(model_dir_string)
            Config = getattr(model_arguments, 'Config')
        except:
            print('Failed to get Config function from ', self.model_dir, '/config.py')
            from crowd_nav.configs.config import Config

        self.config = Config()

        self.test_dir = os.path.join(self.model_dir, self.test_dir)
        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)


        utils.set_seed_everywhere(self.config.env.seed)
        self.device = torch.device("cuda" if self.config.training.cuda and torch.cuda.is_available() else "cpu")
        self.env = utils.make_eval_env(self.config)

        obs_shape = self.env.observation_space.spaces
        action_shape = self.env.action_space.shape
        self.agent = SACAgent(self.config, obs_shape, action_shape, self.device)

        state_dict = torch.load(os.path.join(self.model_dir, 'checkpoints', self.model))
        self.agent.actor.load_state_dict(state_dict)


    def run(self):
        test_size = self.config.env.test_size
        success = 0
        collision = 0
        timeout = 0
        collision_cases = []
        timeout_cases = []

        for i in range(test_size):
            obs = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            stepCounter = 0

            while not done:
                stepCounter = stepCounter + 1
                with utils.eval_mode(self.agent):

                    action = self.agent.act(obs, sample=False)

                next_obs, reward, done, infos = self.env.step(action)


                done = float(done[0])
                episode_reward += reward[0]

                obs = next_obs
            print('Episode', i, 'ends in', stepCounter)
            if isinstance(infos['info'], ReachGoal):
                success += 1
                print('Success')
            elif isinstance(infos['info'], Collision):
                collision += 1
                collision_cases.append(i)
                print('Collision')
            elif isinstance(infos['info'], Timeout):
                timeout += 1
                timeout_cases.append(i)
                print('Time out')
            else:
                raise ValueError('Invalid end signal from environment')

        success_rate = success / test_size
        collision_rate = collision / test_size
        timeout_rate = timeout / test_size
        assert success + collision + timeout == test_size

        print('success rate', success_rate)
        print('collision rate', collision_rate)
        print('timeout rate', timeout_rate)

        result = {}
        result['success_rate'] = success_rate
        result['collision_rate'] = collision_rate
        result['timeout_rate'] = timeout_rate
        result['collision_cases'] = collision_cases
        result['timeout_cases'] = timeout_cases

        np.save(os.path.join(self.test_dir, 'result.npy'), result)


def main():
    workspace = Workspace()
    workspace.run()


if __name__ == '__main__':
    main()
