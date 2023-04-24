import logging
import argparse
import configparser
import os
import torch
import numpy as np
import gym
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA
import hydra
from crowd_sim.envs.utils.action import ActionXY

@hydra.main(config_path='configs/train_NaviSTAR.yaml', strict=True)
def main(cfg):
    work_dir = os.getcwd()
    name_step = '4000'
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--model_dir', type=str, default='il_model.pth')
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--visualize', type=bool,default=True)
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--test_case', type=int, default=1)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--traj', default=False, action='store_true')
    args = parser.parse_args()




    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # configure environment
    env_config_file = r'/home/dinosaur/CrowdNav/crowd_nav/configs/env.config'
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    if args.square:
        env.test_sim = 'square_crossing'
    if args.circle:
        env.test_sim = 'circle_crossing'

    human_num = env_config.getint('sim', 'human_num')
    action_dim = 2
    agent_state_dim = 9
    cfg.agent.params.obs_dim = 5 * human_num + agent_state_dim
    cfg.agent.params.action_dim = action_dim
    cfg.agent.params.action_range = [
        float(-5), float(5)
    ]
    robot = hydra.utils.instantiate(cfg.agent)
    if os.path.isfile('%s/actor_%s.pt' % (work_dir, name_step)):
        print("agent ok")
        robot.load(work_dir, name_step)

    # robot = Robot(env_config, 'robot')
    # robot.set_policy(policy)
    env.set_robot(robot)
    explorer = Explorer(env, robot, device, gamma=0.9)

    robot.print_info()
    if args.visualize:
        ob = env.reset(args.phase, args.test_case)
        ob_arr = []
        for _ob in ob:
            list_ob = list(_ob.__dict__.values())[:5]
            ob_arr = np.concatenate([ob_arr, list_ob], axis=-1)
        agent_ob = list(robot.get_full_state().__dict__.values())[:9]
        ob_arr = np.concatenate([ob_arr, agent_ob], axis=-1)
        ob = ob_arr
        done = False
        last_pos = np.array(robot.get_position())
        while not done:
            action = robot.act(ob)
            trans_action = ActionXY(action[0], action[1])
            logging.info(trans_action)
            ob, _, done, info = env.step(trans_action)
            ob_arr = []
            for _ob in ob:
                list_ob = list(_ob.__dict__.values())[:5]
                ob_arr = np.concatenate([ob_arr, list_ob], axis=-1)
            agent_ob = list(robot.get_full_state().__dict__.values())[:9]
            ob_arr = np.concatenate([ob_arr, agent_ob], axis=-1)
            ob = ob_arr
            logging.info(done)
############################################################################
            # logging.info(ob[0].dis)
############################################################################
            current_pos = np.array(robot.get_position())
            logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
            last_pos = current_pos
        if args.traj:
            env.render('traj', args.video_file)
        else:
            env.render('video', args.video_file)

        logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)
        if robot.visible and info == 'reach goal':
            human_times = env.get_human_times()
            logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))
    else:
        explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)


if __name__ == '__main__':
    main()
