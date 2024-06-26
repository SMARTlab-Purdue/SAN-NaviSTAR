import numpy as np
import torch
import torch.nn as nn
import gym
import argparse
import os
from src.utils import check_reverse, draw
from src.env_wrappers import DummyVecEnv, SubprocVecEnv
from models.model import Policy
import imageio
from crowd_sim.envs.utils.info import *
from crowd_nav.configs.config import Config
from crowd_sim import *


def make_train_env(config):
	def get_env_fn(rank):
		def init_env():
			env = gym.make(config.env.env_name)
			env.configure(config)
			envSeed = config.env.seed + rank if config.env.seed is not None else None
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
	parser = argparse.ArgumentParser('Parse configuration file')
	parser.add_argument('--model_dir', type=str, default='data/navigation/star')
	parser.add_argument('--test_model', type=str, default='00500.pt')
	parser.add_argument('--gif_dir', type=str, default='./gif')
	test_args = parser.parse_args()

	from importlib import import_module
	model_dir_temp = test_args.model_dir
	if model_dir_temp.endswith('/'):
		model_dir_temp = model_dir_temp[:-1]
	try:
		model_dir_string = model_dir_temp.replace('/', '.') + '.configs.config'
		model_arguments = import_module(model_dir_string)
		Config = getattr(model_arguments, 'Config')
	except:
		print('Failed to get Config function from ', test_args.model_dir, '/config.py')
		from crowd_nav.configs.config import Config

	config = Config()

	gif_dir = test_args.gif_dir + '/' + config.robot.policy
	if not os.path.exists(gif_dir):
		os.mkdir(gif_dir)

	torch.manual_seed(config.env.seed)
	torch.cuda.manual_seed_all(config.env.seed)
	if config.training.cuda and torch.cuda.is_available():
		if config.training.cuda_deterministic:
			torch.backends.cudnn.benchmark = False
			torch.backends.cudnn.deterministic = True
		else:
			torch.backends.cudnn.benchmark = True
			torch.backends.cudnn.deterministic = False

	torch.set_num_threads(config.training.num_threads)
	device = torch.device("cuda" if config.training.cuda and torch.cuda.is_available() else "cpu")

	env_name = config.env.env_name
	config.training.num_processes = 1
	envs = make_train_env(config)

	actor_critic = Policy(
		envs.observation_space.spaces,
		envs.action_space,
		base_kwargs=config,
		base=config.robot.policy,
		device=device)

	nn.DataParallel(actor_critic).to(device)

	if test_args.model_dir is not None:
		load_path = os.path.join(test_args.model_dir, 'checkpoints', test_args.test_model)
		print('load path is:', load_path)
		state_dict = torch.load(load_path)
		actor_critic.load_state_dict(state_dict)


	recurrent_cell = 'GRU'
	double_rnn_size = 2 if recurrent_cell == "LSTM" else 1
	num_updates = 10
	done_updates = 0

	while done_updates <= num_updates:
		done = False
		render_obs = envs.reset()
		for key in render_obs.keys():
			render_obs[key] = render_obs[key]
		render_recurrent_hidden_states = {}
		render_recurrent_hidden_states['human_node_rnn'] = np.zeros((config.training.num_processes, 1,
																	   config.SRNN.human_node_rnn_size * double_rnn_size))
		render_recurrent_hidden_states['human_human_edge_rnn'] = np.zeros((config.training.num_processes,
																			 config.sim.human_num + 1,
																			 config.SRNN.human_human_edge_rnn_size * double_rnn_size))
		render_masks = np.zeros((config.training.num_processes, 1))

		frames = []

		while not done:
			_, action, _, render_recurrent_hidden_states = actor_critic.act(
				render_obs, render_recurrent_hidden_states,
				render_masks, deterministic=True)

			image = envs.render()[0]
			frames.append(image)
			robot_traj, robot_goal, human_traj, nav_time = envs.draw()[0]

			action = check_reverse(action)
			render_obs, reward, done, infos = envs.step(action)

			render_masks = np.array([[0.0] if done_ else [1.0] for done_ in done])
			done = done[0, 0]

			for key in render_obs.keys():
				render_obs[key] = render_obs[key]

		if isinstance(infos[0]['info'], ReachGoal):
			status = 'Success'
		elif isinstance(infos[0]['info'], Collision):
			status = 'Collision'
		elif isinstance(infos[0]['info'], Timeout):
			status = 'Timeout'
		else:
			raise ValueError('Invalid end signal from environment')

		final_image = draw(robot_traj, robot_goal, human_traj, status, nav_time)
		frames.append(final_image)
		imageio.mimsave(
			uri="{}/episode{}.gif".format(gif_dir, done_updates),
			ims=frames,
			format="GIF",
			duration=0.1,
			loop=1)

		print(done_updates, status)
		done_updates += 1


if __name__ == '__main__':
	main()
