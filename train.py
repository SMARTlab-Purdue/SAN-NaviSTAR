import sys
import os
import shutil
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import gym
import multiprocessing as mp
from onpolicy.ppo import PPO
from src.utils import *
from src.utils import check_reverse
from src.env_wrappers import DummyVecEnv, SubprocVecEnv
from models.model import Policy
from onpolicy.storage import RolloutStorage
from crowd_nav.configs.config import Config
import crowd_sim


def make_train_env(config):
	def get_env_fn(rank):
		def init_env():
			env = gym.make(config.env.env_name)
			env.configure(config)
			envSeed = config.env.seed + rank if config.env.seed is not None else None
			# environment.render_axis = ax
			env.thisSeed = envSeed
			env.nenv = config.training.num_processes
			if config.training.num_processes > 1:
				env.phase = 'train'
			else:
				env.phase = 'test'
			return env

		return init_env

	if config.training.num_processes == 1:
		return DummyVecEnv([get_env_fn(0)])
	else:
		return SubprocVecEnv([get_env_fn(i) for i in range(
			config.training.num_processes)])


def main():
	config = Config()

	env_name = config.env.env_name
	task = config.env.task
	policy_name = config.robot.policy
	output_dir = os.path.join(config.training.output_dir, task, policy_name)
	# save policy to output_dir
	if os.path.exists(output_dir) and config.training.overwrite: # if I want to overwrite the directory
		shutil.rmtree(output_dir)  # delete an entire directory tree

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	shutil.copytree('crowd_nav/configs', os.path.join(output_dir, 'configs'))

	torch.manual_seed(config.env.seed)
	torch.cuda.manual_seed_all(config.env.seed)
	if config.training.cuda and torch.cuda.is_available():
		if config.training.cuda_deterministic:
			# reproducible but slower
			torch.backends.cudnn.benchmark = False
			torch.backends.cudnn.deterministic = True
		else:
			# not reproducible but faster
			torch.backends.cudnn.benchmark = True
			torch.backends.cudnn.deterministic = False

	torch.set_num_threads(config.training.num_threads)
	device = torch.device("cuda" if config.training.cuda and torch.cuda.is_available() else "cpu")

	# For fastest training: use GRU
	recurrent_cell = 'GRU'

	envs = make_train_env(config)

	actor_critic = Policy(
		envs.observation_space.spaces,
		envs.action_space,
		base_kwargs=config,
		base=config.robot.policy,
		device=device)
	actor_critic.max_action_norm = config.robot.v_pref

	rollouts = RolloutStorage(config.ppo.num_steps,
							  config.training.num_processes,
							  envs.observation_space.spaces,
							  envs.action_space,
							  config.SRNN.human_node_rnn_size,
							  config.SRNN.human_human_edge_rnn_size,
							  recurrent_cell_type=recurrent_cell)

	if config.training.resume: #retrieve the models if resume = True
		load_path = config.training.load_path
		actor_critic.load_state_dict(torch.load(load_path))
		print("Loaded the following checkpoint:", load_path)

	# allow the usage of multiple GPUs to increase the number of examples processed simultaneously
	nn.DataParallel(actor_critic).to(device)

	agent = PPO(
		actor_critic,
		config.ppo.clip_param,
		config.ppo.epoch,
		config.ppo.num_mini_batch,
		config.ppo.value_loss_coef,
		config.ppo.entropy_coef,
		lr=config.training.lr,
		eps=config.training.eps,
		max_grad_norm=config.training.max_grad_norm,
		device=device)

	obs = envs.reset()

	if isinstance(obs, dict):
		for key in obs:
			rollouts.obs[key][0] = obs[key].copy()
	else:
		rollouts.obs[0] = obs.copy()

	episode_rewards = deque(maxlen=100)

	start = time.time()
	num_updates = int(config.training.num_env_steps) // config.ppo.num_steps // config.training.num_processes

	for j in range(num_updates):
		if config.training.use_linear_lr_decay:
			update_linear_schedule(
				agent.optimizer, j, num_updates, config.training.lr)
		for step in range(config.ppo.num_steps):
			# Sample actions
			with torch.no_grad():
				rollouts_obs = {}
				for key in rollouts.obs:
					rollouts_obs[key] = rollouts.obs[key][step]
				rollouts_hidden_s = {}
				for key in rollouts.recurrent_hidden_states:
					rollouts_hidden_s[key] = rollouts.recurrent_hidden_states[key][step]
				value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
					rollouts_obs, rollouts_hidden_s,
					rollouts.masks[step])

				value = check_reverse(value)
				action = check_reverse(action)
				action_log_prob = check_reverse(action_log_prob)
				for key in recurrent_hidden_states.keys():
					recurrent_hidden_states[key] = check_reverse(recurrent_hidden_states[key])

			# Obser reward and next obs
			obs, reward, done, infos = envs.step(action)

			for info in infos:
				# print(info.keys())
				if 'episode_reward' in info.keys():
					episode_rewards.append(info['episode_reward'])

			# If done then clean the history of observations.
			masks = np.array([[0.0] if done_ else [1.0] for done_ in done])
			bad_masks = np.array([[0.0] if 'bad_transition' in info.keys() else [1.0]
				 for info in infos])


			rollouts.insert(obs, recurrent_hidden_states, action,
						action_log_prob, value, reward, masks, bad_masks)

		with torch.no_grad():
			rollouts_obs = {}
			for key in rollouts.obs:
				rollouts_obs[key] = rollouts.obs[key][-1]
			rollouts_hidden_s = {}
			for key in rollouts.recurrent_hidden_states:
				rollouts_hidden_s[key] = rollouts.recurrent_hidden_states[key][-1]
			next_value = actor_critic.get_value(
				rollouts_obs, rollouts_hidden_s,
				rollouts.masks[-1]).detach()
			next_value = check_reverse(next_value)

		rollouts.compute_returns(next_value, config.ppo.use_gae, config.reward.gamma,
								 config.ppo.gae_lambda, config.training.use_proper_time_limits)

		value_loss, action_loss, dist_entropy = agent.update(rollouts)
		rollouts.after_update()


		# save the models for every interval-th episode or for the last epoch
		if (j % config.training.save_interval == 0
			or j == num_updates - 1) :
			save_path = os.path.join(output_dir, 'checkpoints')
			if not os.path.exists(save_path):
				os.mkdir(save_path)

			torch.save(actor_critic.state_dict(), os.path.join(save_path, '%.5i' % j + ".pt"))

		if j % config.training.log_interval == 0 and len(episode_rewards) > 1:
			total_num_steps = (j + 1) * config.training.num_processes * config.ppo.num_steps
			end = time.time()
			print(
				"Updates {}/{}, num timesteps {}, FPS {} policy {} seed {}\n "
				"Last {} training episodes: mean/median reward "
				"{:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
					.format(j, num_updates, total_num_steps,
							int(total_num_steps / (end - start)),
							policy_name, config.env.seed,
							len(episode_rewards), np.mean(episode_rewards),
							np.median(episode_rewards), np.min(episode_rewards),
							np.max(episode_rewards), dist_entropy, value_loss,
							action_loss))

			df = pd.DataFrame({'misc/nupdates': [j], 'misc/total_timesteps': [total_num_steps],
							   'fps': int(total_num_steps / (end - start)), 'eprewmean': [np.mean(episode_rewards)],
							   'loss/policy_entropy': dist_entropy, 'loss/policy_loss': action_loss,
							   'loss/value_loss': value_loss})

			if os.path.exists(os.path.join(output_dir, 'progress.csv')) and j > 20:
				df.to_csv(os.path.join(output_dir, 'progress.csv'), mode='a', header=False, index=False)
			else:
				df.to_csv(os.path.join(output_dir, 'progress.csv'), mode='w', header=True, index=False)


	envs.close()


if __name__ == '__main__':
	mp.set_start_method('spawn')
	main()
