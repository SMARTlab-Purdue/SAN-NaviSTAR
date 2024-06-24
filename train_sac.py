import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
import shutil
from offpolicy.video import VideoRecorder
from offpolicy.logger import Logger
from offpolicy.replay_buffer import ReplayBuffer
from offpolicy.agent.sac import SACAgent
import offpolicy.utils as utils
from crowd_nav.configs.config import Config
import crowd_sim
import os
from collections import deque

class Workspace(object):
    def __init__(self):

        self.config = Config()
        utils.set_seed_everywhere(self.config.env.seed)

        env_name = self.config.env.env_name
        task = self.config.env.task
        policy_name = self.config.robot.policy + '_sac'
        self.output_dir = os.path.join(self.config.training.output_dir, task, policy_name)
        # save policy to output_dir
        if os.path.exists(self.output_dir) and self.config.training.overwrite:  # if I want to overwrite the directory
            shutil.rmtree(self.output_dir)  # delete an entire directory tree

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        shutil.copytree('crowd_nav/configs', os.path.join(self.output_dir, 'configs'))

        self.model_dir = os.path.join(self.output_dir, 'checkpoints')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.work_dir = self.output_dir
        print(f'workspace: {self.work_dir}')

        self.logger = Logger(self.work_dir,
                             save_tb=True,
                             log_frequency=10000,
                             agent='sac')


        self.device = torch.device("cuda" if self.config.training.cuda and torch.cuda.is_available() else "cpu")
        self.env = utils.make_env(self.config)
        self.eval_env = utils.make_eval_env(self.config)

        obs_shape = self.env.observation_space.spaces
        action_shape = self.env.action_space.shape
        self.agent = SACAgent(self.config, obs_shape, action_shape, self.device)

        self.replay_buffer = ReplayBuffer(obs_shape,
                                          action_shape,
                                          int(self.config.sac.num_train_steps),
                                          self.device)

        self.video_recorder = VideoRecorder(
            self.work_dir if self.config.sac.save_video else None)
        self.step = 0
        self.interaction = 0

        self.max_success_rate = 0.

    def evaluate(self):
        success = 0
        collision = 0
        timeout = 0
        average_episode_reward = 0
        for episode in range(self.config.sac.num_eval_episodes):
            self.eval_env.case_counter['test'] = 0
            obs = self.eval_env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, info = self.eval_env.step(action)
                done = done[0]
                episode_reward += reward[0]

            average_episode_reward += episode_reward
            status = str(info['info'])
            if status == 'Reaching goal':
                success += 1
            elif status == 'Collision':
                collision += 1
            elif status == 'Timeout':
                timeout += 1
        average_episode_reward /= self.config.sac.num_eval_episodes
        success_rate = success / self.config.sac.num_eval_episodes
        collision_rate = collision / self.config.sac.num_eval_episodes
        timeout_rate = timeout / self.config.sac.num_eval_episodes

        if success_rate > self.max_success_rate:
            self.max_success_rate = success_rate
            torch.save(self.agent.actor.state_dict(), os.path.join(self.model_dir, './best_sac_actor.pt'))

        print('eval', average_episode_reward)
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/success_rate', success_rate,
                        self.step)
        self.logger.log('eval/collision_rate', collision_rate,
                        self.step)
        self.logger.log('eval/timeout_rate', timeout_rate,
                        self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_step, episode_reward, done = 0, 0, 0, True
        start_time = time.time()
        episode_rewards = deque(maxlen=100)
        reward_list = []
        while self.step < self.config.sac.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.config.sac.num_seed_steps))


                if self.interaction > self.config.sac.save_interval:
                    self.interaction = 0
                    filename = 'sac_actor' + str(self.step) + '.pt'
                    torch.save(self.agent.actor.state_dict(), os.path.join(self.model_dir, filename))
                    np.save(os.path.join(self.output_dir, 'reward.npy'), reward_list)
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)
                if self.step >= self.config.sac.num_seed_steps:
                    episode_rewards.append(episode_reward)
                    reward_list.append(np.mean(episode_rewards))
                    print('%d/%d, %d, %f' % (self.step, self.config.sac.num_train_steps, episode_step, np.mean(episode_rewards)))

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.config.sac.num_seed_steps:
                action = self.env.action_space.sample()
                action = utils.clip_action(action, clip_norm=True, max_norm=self.config.robot.v_pref)
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)
                    action = utils.clip_action(action, clip_norm=True, max_norm=self.config.robot.v_pref)

            # run training update
            if self.step >= self.config.sac.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)


            next_obs, reward, done, info = self.env.step(action)

            # allow infinite bootstrap
            done = float(done[0])
            done_no_max = done
            episode_reward += reward[0]

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            self.interaction += 1



def main():
    workspace = Workspace()
    workspace.run()


if __name__ == '__main__':
    main()
