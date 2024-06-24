import argparse
import os
import sys
import torch
import torch.nn as nn
import gym
from src.env_wrappers import DummyVecEnv, SubprocVecEnv
from src.utils import check_reverse
from crowd_sim import *
from crowd_sim.envs.utils.info import *
from models.model import Policy
import numpy as np



def make_test_env(config):
	def get_env_fn(rank):
		def init_env():
			env = gym.make(config.env.env_name)
			env.configure(config)
			envSeed = config.env.seed + rank if config.env.seed is not None else None
			env.thisSeed = envSeed
			env.nenv = config.testing.num_processes
			env.phase = 'test'
			return env

		return init_env

	if config.testing.num_processes == 1:
		return DummyVecEnv([get_env_fn(0)])
	else:
		return SubprocVecEnv([get_env_fn(i) for i in range(
			config.testing.num_processes)])

def main():
	parser = argparse.ArgumentParser('Parse configuration file')
	parser.add_argument('--model_dir', type=str, default='data/navigation/star')
	# if -1, it will run 500 different cases; if >=0, it will run the specified test case repeatedly
	parser.add_argument('--test_case', type=int, default=-1)
	parser.add_argument('--test_model', type=str, default='00500.pt')
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

	log_file = os.path.join(test_args.model_dir, 'test')
	if not os.path.exists(log_file):
		os.mkdir(log_file)

	torch.manual_seed(config.env.seed)
	torch.cuda.manual_seed_all(config.env.seed)
	if config.training.cuda:
		if config.training.cuda_deterministic:
			torch.backends.cudnn.benchmark = False
			torch.backends.cudnn.deterministic = True
		else:
			torch.backends.cudnn.benchmark = True
			torch.backends.cudnn.deterministic = False

	torch.set_num_threads(1)
	device = torch.device("cuda" if config.training.cuda else "cpu")


	load_path = os.path.join(test_args.model_dir, 'checkpoints', test_args.test_model)
	print('load path is:', load_path)

	eval_dir = os.path.join(test_args.model_dir, 'eval')
	if not os.path.exists(eval_dir):
		os.mkdir(eval_dir)

	envs = make_test_env(config)

	actor_critic = Policy(
		envs.observation_space.spaces,
		envs.action_space,
		base_kwargs=config,
		base=config.robot.policy,
		device=device)

	actor_critic.load_state_dict(torch.load(load_path, map_location=device))
	actor_critic.base.nenv = config.testing.num_processes

	nn.DataParallel(actor_critic).to(device)

	test_size = config.env.test_size
	recurrent_cell = 'GRU'
	double_rnn_size = 2 if recurrent_cell == "LSTM" else 1

	obs = envs.reset()
	for key in obs.keys():
		obs[key] = obs[key]
	rewards = []
	success = 0
	collision = 0
	timeout = 0
	collision_cases = []
	timeout_cases = []

	for k in range(test_size):
		done = False
		stepCounter = 0

		eval_recurrent_hidden_states = {}
		eval_recurrent_hidden_states['human_node_rnn'] = np.zeros((config.testing.num_processes, 1,
																	 config.SRNN.human_node_rnn_size * double_rnn_size))
		eval_recurrent_hidden_states['human_human_edge_rnn'] = np.zeros((config.testing.num_processes,
																		   config.sim.human_num + 1,
																		   config.SRNN.human_human_edge_rnn_size * double_rnn_size))
		eval_masks = np.zeros((config.testing.num_processes, 1))

		while not done:
			stepCounter = stepCounter + 1
			with torch.no_grad():
				_, action, _, eval_recurrent_hidden_states = actor_critic.act(
					obs,
					eval_recurrent_hidden_states,
					eval_masks,
					deterministic=True)

			action = check_reverse(action)
			obs, rew, done, infos = envs.step(action)

			rewards.append(rew[0, 0])

			eval_masks = np.array([[0.0] if done_ else [1.0] for done_ in done])
			done = done[0, 0]

		print('Episode', k, 'ends in', stepCounter)
		if isinstance(infos[0]['info'], ReachGoal):
			success += 1
			print('Success')
		elif isinstance(infos[0]['info'], Collision):
			collision += 1
			collision_cases.append(k)
			print('Collision')
		elif isinstance(infos[0]['info'], Timeout):
			timeout += 1
			timeout_cases.append(k)
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

	np.save(os.path.join(log_file, 'result.npy'), result)


if __name__ == '__main__':
	main()
