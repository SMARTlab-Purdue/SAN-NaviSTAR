#!/usr/bin/env python3
import numpy as np
import torch
import os
import gym
import time
import configparser
from crowd_sim.envs.utils.action import ActionXY
from logger import Logger
from replay_buffer import ReplayBuffer
from reward_model import RewardModel
from collections import deque
import csv
import utils_a as utils
import hydra
print(os.getcwd())


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        self.buffer_dir = "/home/peter/CrowdNav/crowd_nav/buffer_data/"
        self.name_step = '4000'
        self.buffer_name = self.name_step + 'actions.csv'
        self.buffer_path = self.buffer_dir + self.buffer_name
        self.reward_data_path = "/home/peter/CrowdNav/crowd_nav/reward_model_data/"
        self.seed_sample_flag = True

        self.cfg = cfg
        self.agent_state_dim = 9
        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.log_success = False
        env_config = configparser.RawConfigParser()
        env_config.read('/home/peter/CrowdNav/crowd_nav/configs/env.config')
        self.env = gym.make('CrowdSim-v0')
        self.env.configure(env_config)
        self.human_num = env_config.getint('sim', 'human_num')
        self.action_dim = 2
        cfg.agent.params.obs_dim = 5 * self.human_num + self.agent_state_dim
        cfg.agent.params.action_dim = self.action_dim
        cfg.agent.params.action_range = [
            float(-5), float(5)
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)
        self.env.set_robot(self.agent)

        self.replay_buffer = ReplayBuffer(
            (5 * self.human_num + self.agent_state_dim,), (self.action_dim,),
            int(cfg.replay_buffer_capacity),
            self.device)
        if os.path.isfile(self.buffer_path):
            print("buffer ok")
            self.replay_buffer.load(self.buffer_dir + str(self.name_step))
        if os.path.isfile('%s/actor_%s.pt' % (self.work_dir, self.name_step)):
            print("agent ok")
            self.seed_sample_flag = False
            self.agent.load(self.work_dir, self.name_step)

        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        self.reward_model = RewardModel(
            5 * self.human_num + self.agent_state_dim, self.action_dim,
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation,
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch,
            large_batch=cfg.large_batch,
            label_margin=cfg.label_margin,
            teacher_beta=cfg.teacher_beta,
            teacher_gamma=cfg.teacher_gamma,
            teacher_eps_mistake=cfg.teacher_eps_mistake,
            teacher_eps_skip=cfg.teacher_eps_skip,
            teacher_eps_equal=cfg.teacher_eps_equal)
        if os.path.isfile('%s/reward_model_%s_0.pt' % (self.work_dir, self.name_step)):
            print("reward model parameter ok")
            self.reward_model.load(self.work_dir, self.name_step)
        else:
            print("no reward model weight exist!")

    def evaluate(self):
        average_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0

        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            true_episode_reward = 0
            if self.log_success:
                episode_success = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, extra = self.env.step(action)

                episode_reward += reward
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])

            average_episode_reward += episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success

        average_episode_reward /= self.cfg.num_eval_episodes
        average_true_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0

        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/true_episode_reward', average_true_episode_reward,
                        self.step)
        if self.log_success:
            self.logger.log('eval/success_rate', success_rate,
                            self.step)
            self.logger.log('train/true_episode_success', success_rate,
                            self.step)
        self.logger.dump(self.step)

    def learn_reward(self, first_flag=0):

        # get feedbacks
        labeled_queries, noisy_queries = 0, 0
        if first_flag == 1:
            # if it is first time to get feedback, need to use random sampling
            labeled_queries = self.reward_model.uniform_sampling(self.env)
        else:
            if self.cfg.feed_type == 0:
                labeled_queries = self.reward_model.uniform_sampling(self.env)
            elif self.cfg.feed_type == 1:
                labeled_queries = self.reward_model.disagreement_sampling()
            elif self.cfg.feed_type == 2:
                labeled_queries = self.reward_model.entropy_sampling()
            elif self.cfg.feed_type == 3:
                labeled_queries = self.reward_model.kcenter_sampling()
            elif self.cfg.feed_type == 4:
                labeled_queries = self.reward_model.kcenter_disagree_sampling()
            elif self.cfg.feed_type == 5:
                labeled_queries = self.reward_model.kcenter_entropy_sampling()
            else:
                raise NotImplementedError

        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries

        train_acc = 0
        if self.labeled_feedback > 0:
            # update reward
            for epoch in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0:
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    train_acc = self.reward_model.train_reward()
                total_acc = np.mean(train_acc)

                if total_acc > 0.97:
                    break;

        print("Reward function is updated!! ACC: " + str(total_acc))

    def save_avg_train_true_return(self, save_path, avg_train_true_return):
        save_path = save_path + "avg_train_true_return.csv"
        with open(save_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(avg_train_true_return)

    def run(self):
        episode, episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0

        # store train returns of recent 10 episodes
        avg_train_true_return = deque([], maxlen=10)
        start_time = time.time()
        self.env.reset()
        interact_count = 0
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)

                if self.log_success:
                    self.logger.log('train/episode_success', episode_success,
                                    self.step)
                    self.logger.log('train/true_episode_success', episode_success,
                                    self.step)
                obs = self.env.reset()
                ob_orca = obs.copy()
                ob_arr = []
                for ob in obs:
                    list_ob = list(ob.__dict__.values())[:5]
                    ob_arr = np.concatenate([ob_arr, list_ob], axis=-1)
                agent_ob = list(self.agent.get_full_state().__dict__.values())[:9]
                ob_arr = np.concatenate([ob_arr, agent_ob], axis=-1)
                obs = ob_arr
                self.agent.reset()
                done = False
                episode_reward = 0
                avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps/10 and self.seed_sample_flag:
                action = self.env.action_space.sample()
            elif self.step < self.cfg.num_seed_steps and self.seed_sample_flag:
                action = self.agent.act_orca(ob_orca)
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)
            if self.step > self.cfg.num_seed_steps or not self.seed_sample_flag:
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step,
                                            gradient_update=1, K=self.cfg.topK)

            trans_action = ActionXY(action[0], action[1])
            next_obs, reward, done, extra = self.env.step(trans_action)
            ob_orca = next_obs.copy()
            n_ob_arr = []
            for ob in next_obs:
                list_ob = list(ob.__dict__.values())[:5]
                n_ob_arr = np.concatenate([n_ob_arr, list_ob], axis=-1)
            agent_ob = list(self.agent.get_full_state().__dict__.values())[:9]
            n_ob_arr = np.concatenate([n_ob_arr, agent_ob], axis=-1)
            next_obs = n_ob_arr

            all = np.concatenate([obs, action], axis=-1)

            reward_hat = self.reward_model.r_hat(all)

            # allow infinite bootstrap
            done = float(done)
            # done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            done_no_max = 0 if episode_step + 1 == 400 else done
            episode_reward += reward_hat
            true_episode_reward += reward

            if self.log_success:
                episode_success = max(episode_success, extra['success'])

            # adding data to the reward training data
            self.reward_model.add_data(obs, action, reward, done)
            self.replay_buffer.add(
                obs, action, reward_hat,
                next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1

        self.save_avg_train_true_return(self.reward_data_path, avg_train_true_return)
        self.agent.save(self.work_dir, self.step)
        self.reward_model.save(self.work_dir, self.step)
        self.reward_model.save_reward_data(self.reward_data_path)
        self.replay_buffer.save(self.buffer_dir + str(self.step))


@hydra.main(config_path='configs/train_NaviSTAR.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
