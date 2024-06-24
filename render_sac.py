import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import offpolicy.utils as utils
import crowd_sim
import os
from offpolicy.agent.sac import SACAgent
import imageio
from crowd_sim.envs.utils.info import *
from src.utils import draw

class Workspace(object):
    def __init__(self):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.model_dir = 'data/navigation/star_sac'
        self.gif_dir = './gif'
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

        self.gif_dir = os.path.join(self.gif_dir, self.config.robot.policy)
        if not os.path.exists(self.gif_dir):
            os.mkdir(self.gif_dir)


        utils.set_seed_everywhere(self.config.env.seed)
        self.device = torch.device("cuda" if self.config.training.cuda and torch.cuda.is_available() else "cpu")
        self.env = utils.make_eval_env(self.config)

        obs_shape = self.env.observation_space.spaces
        action_shape = self.env.action_space.shape
        self.agent = SACAgent(self.config, obs_shape, action_shape, self.device)

        state_dict = torch.load(os.path.join(self.model_dir, 'checkpoints', self.model))
        self.agent.actor.load_state_dict(state_dict)


    def run(self):
        render_episodes = 50

        for i in range(render_episodes):
            obs = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            frames = []

            while not done:
                with utils.eval_mode(self.agent):

                    action = self.agent.act(obs, sample=False)

                image = self.env.render()
                frames.append(image)
                robot_traj, robot_goal, human_traj, nav_time = self.env.draw()

                next_obs, reward, done, infos = self.env.step(action)

                done = float(done[0])
                episode_reward += reward[0]

                obs = next_obs

            if isinstance(infos['info'], ReachGoal):
                status = 'Success'
            elif isinstance(infos['info'], Collision):
                status = 'Collision'
            elif isinstance(infos['info'], Timeout):
                status = 'Timeout'
            else:
                raise ValueError('Invalid end signal from environment')

            print(i, status)

            final_image = draw(robot_traj, robot_goal, human_traj, status, nav_time)
            frames.append(final_image)
            imageio.mimsave(
                uri="{}/episode{}.gif".format(self.gif_dir, i),
                ims=frames,
                format="GIF",
                duration=0.1,
                loop=1)

def main():
    workspace = Workspace()
    workspace.run()


if __name__ == '__main__':
    main()
