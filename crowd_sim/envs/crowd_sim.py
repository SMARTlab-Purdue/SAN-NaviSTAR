import random
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.policy.orca import ORCA
from crowd_nav.policy.social_force import SOCIAL_FORCE
from crowd_sim.envs.utils.state import *
from crowd_nav.policy.policy_factory import policy_factory
import gym
import numpy as np
from numpy.linalg import norm
from crowd_sim.envs.utils.info import *


class CrowdSim(gym.Env):
    def __init__(self):
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None

        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_dist_front = None
        self.discomfort_penalty_factor = None

        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None

        self.circle_radius = None
        self.human_num = None

        self.action_space=None
        self.observation_space=None

        # limit FOV
        self.robot_fov = None
        self.human_fov = None

        self.dummy_human = None
        self.dummy_robot = None

        #seed
        self.thisSeed=None

        #nenv
        self.nenv=None
        self.phase=None
        self.test_case=None

        self.humans = []
        self.potential = None

    # configurate the environment with the given config
    def configure(self, config):
        self.config = config

        self.task = config.env.task
        self.time_limit = config.env.time_limit
        self.time_step = config.env.time_step
        self.randomize_attributes = config.env.randomize_attributes
        self.visible_dis = config.robot.visible_dis

        self.success_reward = config.reward.success_reward
        self.follow_reward = config.reward.follow_reward
        self.collision_penalty = config.reward.collision_penalty
        self.discomfort_dist = config.reward.discomfort_dist_back
        self.discomfort_dist_front = config.reward.discomfort_dist_front
        self.discomfort_penalty_factor = config.reward.discomfort_penalty_factor

        self._max_episode_steps = int((config.env.time_limit - 1) // config.env.time_step + 1)


        if self.config.humans.policy == 'orca' or self.config.humans.policy == 'social_force':
            self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
            self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': self.config.env.val_size,
                              'test': self.config.env.test_size}
            self.circle_radius = config.sim.circle_radius
            self.human_num = config.sim.human_num
            self.static_human_num = config.sim.static_human_num
            self.group_human = config.sim.group_human

        else:
            raise NotImplementedError
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        self.robot_fov = np.pi * config.robot.FOV
        self.human_fov = np.pi * config.humans.FOV

        # set dummy human and dummy robot
        # dummy humans, used if any human is not in view of other agents
        self.dummy_human = Human(self.config, 'humans')
        self.dummy_human.set(15, 15, 15, 15, 0, 0, 0)
        self.dummy_human.time_step = config.env.time_step

        self.dummy_robot = Robot(self.config, 'robot')
        self.dummy_robot.set(15, 15, 15, 15, 0, 0, 0)
        self.dummy_robot.time_step = config.env.time_step
        self.dummy_robot.kinematics = 'holonomic'
        self.dummy_robot.policy = SOCIAL_FORCE(config)


        # configure noise in state
        self.add_noise = config.noise.add_noise
        if self.add_noise:
            self.noise_type = config.noise.type
            self.noise_magnitude = config.noise.magnitude


        # configure randomized goal changing of humans midway through episode
        self.random_goal_changing = config.humans.random_goal_changing
        if self.random_goal_changing:
            self.goal_change_chance = config.humans.goal_change_chance

        # configure randomized goal changing of humans after reaching their respective goals
        self.end_goal_changing = config.humans.end_goal_changing
        if self.end_goal_changing:
            self.end_goal_change_chance = config.humans.end_goal_change_chance

        # configure randomized radii changing when reaching goals
        self.random_radii = config.humans.random_radii

        # configure randomized v_pref changing when reaching goals
        self.random_v_pref = config.humans.random_v_pref

        # configure randomized goal changing of humans after reaching their respective goals
        self.random_unobservability = config.humans.random_unobservability
        if self.random_unobservability:
            self.unobservable_chance = config.humans.unobservable_chance


        self.last_human_states = np.zeros((self.human_num, 5))

        # configure randomized policy changing of humans every episode
        self.random_policy_changing = config.humans.random_policy_changing

        # set robot for this envs
        rob_RL = Robot(config, 'robot')
        self.set_robot(rob_RL)

    def set_robot(self, robot):
        self.robot = robot

        d = {}
        # [px, py, r, gx, gy, v_pref, theta]
        d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 7,), dtype=np.float32)
        # [vx, vy]
        d['temporal_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2,), dtype=np.float32)
        # [[x1-px, y1-py],
        #  [x2-px, y2-py],
        #        ...
        #  [xi-px, yi-py]]
        d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.human_num, 2), dtype=np.float32)

        # see goal as a static human
        # goal's one-hot encode is different from human
        # [[x1-px, y1-py, 0, 1],
        #  [x2-px, y2-py, 0, 1],
        #        ...
        #  [xi-px, yi-py, 0, 1],
        #  [gx-px, gy-py, 1, 0]]
        d['spatial_edges_transformer'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.human_num + 1, 4),
                                                        dtype=np.float32)

        # if human is too far from robot, do not consider it's state
        # the last element is the always 1
        # [1, 1, 0, 0, 0, ..., 0, 1]
        d['visible_masks'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.human_num + 1,),
                                            dtype=np.float32)
        # [px, py, 1, 0]
        d['robot_pos'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 4),
                                        dtype=np.float32)

        d['flatten_obs'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.human_num * 2 + 9,),
                                          dtype=np.float32)

        self.observation_space = gym.spaces.Dict(d)

        high = np.inf * np.ones([2, ])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def generate_random_human_position(self, human_num):
        for i in range(human_num):
            self.humans.append(self.generate_circle_crossing_human())

    def generate_static_human_position(self, human_num):
        for i in range(human_num):
            self.humans.append(self.generate_static_human())


    def generate_circle_static_obstacle(self, position=None):
        human = Human(self.config, 'humans')
        # For fixed position
        if position:
            px, py = position
        # For random position
        else:
            while True:
                angle = np.random.random() * np.pi * 2
                # add some noise to simulate all the possible cases robot could meet with human
                v_pref = 1.0 if human.v_pref == 0 else human.v_pref
                px_noise = (np.random.random() - 0.5) * v_pref
                py_noise = (np.random.random() - 0.5) * v_pref
                px = self.circle_radius * np.cos(angle) + px_noise
                py = self.circle_radius * np.sin(angle) + py_noise
                collide = False
                for agent in [self.robot] + self.humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                    if norm((px - agent.px, py - agent.py)) < min_dist or \
                                    norm((px - agent.gx, py - agent.gy)) < min_dist:
                        collide = True
                        break
                if not collide:
                    break

        # make it a static obstacle
        # px, py, gx, gy, vx, vy, theta
        human.set(px, py, px, py, 0, 0, 0, v_pref=0)
        return human


    def generate_circle_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()

        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            v_pref = 1.0 if human.v_pref == 0 else human.v_pref
            px_noise = (np.random.random() - 0.5) * v_pref
            py_noise = (np.random.random() - 0.5) * v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False

            if self.group_human:
                collide = self.check_collision_group((px, py), human.radius)

            else:
                for i, agent in enumerate([self.robot] + self.humans):
                    # keep human at least 3 meters away from robot
                    if self.robot.kinematics == 'unicycle' and i == 0:
                        min_dist = self.circle_radius / 2 # Todo: if circle_radius <= 4, it will get stuck here
                    else:
                        min_dist = human.radius + agent.radius + self.discomfort_dist
                    if norm((px - agent.px, py - agent.py)) < min_dist or \
                            norm((px - agent.gx, py - agent.gy)) < min_dist:
                        collide = True
                        break
            if not collide:
                break

        human.set(px, py, -px, -py, 0, 0, 0)
        return human


    def generate_static_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        human.isObstacle = True
        while True:
            px, py = np.random.uniform(-(self.circle_radius - 1), self.circle_radius - 1, 2)
            collide = False

            if self.group_human:
                collide = self.check_collision_group((px, py), human.radius)

            else:
                for i, agent in enumerate([self.robot] + self.humans):
                    # keep human at least 3 meters away from robot
                    if self.robot.kinematics == 'unicycle' and i == 0:
                        min_dist = self.circle_radius / 2
                    else:
                        min_dist = human.radius + agent.radius + self.discomfort_dist
                    if norm((px - agent.px, py - agent.py)) < min_dist or \
                            norm((px - agent.gx, py - agent.gy)) < min_dist:
                        collide = True
                        break
            if not collide:
                break

        human.set(px, py, px, py, 0, 0, 0, v_pref=0)
        return human

    # add noise according to env.config to observation
    def apply_noise(self, ob):
        if isinstance(ob[0], ObservableState):
            for i in range(len(ob)):
                if self.noise_type == 'uniform':
                    noise = np.random.uniform(-self.noise_magnitude, self.noise_magnitude, 5)
                elif self.noise_type == 'gaussian':
                    noise = np.random.normal(size=5)
                else:
                    print('noise type not defined')
                ob[i].px = ob[i].px + noise[0]
                ob[i].py = ob[i].px + noise[1]
                ob[i].vx = ob[i].px + noise[2]
                ob[i].vy = ob[i].px + noise[3]
                ob[i].radius = ob[i].px + noise[4]
            return ob
        else:
            if self.noise_type == 'uniform':
                noise = np.random.uniform(-self.noise_magnitude, self.noise_magnitude, len(ob))
            elif self.noise_type == 'gaussian':
                noise = np.random.normal(size = len(ob))
            else:
                print('noise type not defined')
                noise = [0] * len(ob)

            return ob + noise


    # update the robot belief of human states
    # if a human is visible, its state is updated to its current ground truth state
    # else we assume it keeps going in a straight line with last observed velocity
    def update_last_human_states(self, human_visibility, reset):
        """
        update the self.last_human_states array
        human_visibility: list of booleans returned by get_human_in_fov (e.x. [T, F, F, T, F])
        reset: True if this function is called by reset, False if called by step
        :return:
        """
        # keep the order of 5 humans at each timestep
        for i in range(self.human_num):
            if human_visibility[i]:
                humanS = np.array(self.humans[i].get_observable_state_list())
                self.last_human_states[i, :] = humanS

            else:
                if reset:
                    humanS = np.array([15., 15., 0., 0., 0.3])
                    self.last_human_states[i, :] = humanS

                else:
                    px, py, vx,  vy, theta = self.last_human_states[i, :]
                    # Plan A: linear approximation of human's next position
                    px = px + vx * self.time_step
                    py = py + vy * self.time_step
                    self.last_human_states[i, :] = np.array([px, py, vx, vy, theta])

                    # Plan B: assume the human doesn't move, use last observation
                    # self.last_human_states[i, :] = np.array([px, py, 0., 0., r])


    # return the ground truth locations of all humans
    def get_true_human_states(self):
        true_human_states = np.zeros((self.human_num, 2))
        for i in range(self.human_num):
            humanS = np.array(self.humans[i].get_observable_state_list())
            true_human_states[i, :] = humanS[:2]
        return true_human_states


    def randomize_human_policies(self):
        """
        Randomize the moving humans' policies to be either orca or social force
        """
        for human in self.humans:
            if not human.isObstacle:
                new_policy = random.choice(['orca', 'social_force'])
                new_policy = policy_factory[new_policy](self.config)
                human.set_policy(new_policy)


    # Generates group of circum_num humans in a circle formation at a random viable location
    def generate_circle_group_obstacle(self, circum_num):
        group_circumference = self.config.humans.radius * 2 * circum_num
        # print("group circum: ", group_circumference)
        group_radius = group_circumference / (2 * np.pi)
        # print("group radius: ", group_radius)
        while True:
            rand_cen_x = np.random.uniform(-3, 3)
            rand_cen_y = np.random.uniform(-3, 3)
            success = True
            for i, group in enumerate(self.circle_groups):
                # print(i)
                dist_between_groups = np.sqrt((rand_cen_x - group[1]) ** 2 + (rand_cen_y - group[2]) ** 2)
                sum_radius = group_radius + group[0] + 2 * self.config.humans.radius
                if dist_between_groups < sum_radius:
                    success = False
                    break
            if success:
                # print("------------\nsuccessfully found valid x: ", rand_cen_x, " y: ", rand_cen_y)
                break
        self.circle_groups.append((group_radius, rand_cen_x, rand_cen_y))

        arc = 2 * np.pi / circum_num
        for i in range(circum_num):
            angle = arc * i
            curr_x = rand_cen_x + group_radius * np.cos(angle)
            curr_y = rand_cen_y + group_radius * np.sin(angle)
            point = (curr_x, curr_y)
            # print("adding circle point: ", point)
            curr_human = self.generate_circle_static_obstacle(point)
            curr_human.isObstacle = True
            self.humans.append(curr_human)

        return

    # given an agent's xy location and radius, check whether it collides with all other humans
    # including the circle group and moving humans
    # return True if collision, False if no collision
    def check_collision_group(self, pos, radius):
        # check circle groups
        for r, x, y in self.circle_groups:
            if np.linalg.norm([pos[0] - x, pos[1] - y]) <= r + radius + 2 * 0.5: # use 0.5 because it's the max radius of human
                return True

        # check moving humans
        for human in self.humans:
            if human.isObstacle:
                pass
            else:
                if np.linalg.norm([pos[0] - human.px, pos[1] - human.py]) <= human.radius + radius:
                    return True
        return False

    # check collision between robot goal position and circle groups
    def check_collision_group_goal(self, pos, radius):
        # print('check goal', len(self.circle_groups))
        collision = False
        # check circle groups
        for r, x, y in self.circle_groups:
            if np.linalg.norm([pos[0] - x, pos[1] - y]) <= r + radius + 4 * 0.5: # use 0.5 because it's the max radius of human
                collision = True
        return collision



    # set robot initial state and generate all humans for reset function
    # for crowd nav: human_num == self.human_num
    # for leader follower: human_num = self.human_num - 1
    def generate_robot_humans(self, phase, human_num=None):
        if human_num is None:
            human_num = self.human_num
        # for Group environment
        if self.group_human:
            # set the robot in a dummy far away location to avoid collision with humans
            self.robot.set(15, 15, 15, 15, 0, 0, np.pi / 2)

            # generate humans
            self.circle_groups = []
            humans_left = human_num

            while humans_left > 0:
                # print("****************\nhumans left: ", humans_left)
                if humans_left <= 4:
                    if phase in ['train', 'val']:
                        self.generate_random_human_position(human_num=humans_left)
                    else:
                        self.generate_random_human_position(human_num=humans_left)
                    humans_left = 0
                else:
                    if humans_left < 10:
                        max_rand = humans_left
                    else:
                        max_rand = 10
                    # print("randint from 4 to ", max_rand)
                    circum_num = np.random.randint(4, max_rand)
                    # print("circum num: ", circum_num)
                    self.generate_circle_group_obstacle(circum_num)
                    humans_left -= circum_num

            # randomize starting position and goal position while keeping the distance of goal to be > 6
            # set the robot on a circle with radius 5.5 randomly
            rand_angle = np.random.uniform(0, np.pi * 2)
            # print('rand angle:', rand_angle)
            increment_angle = 0.0
            while True:
                px_r = np.cos(rand_angle + increment_angle) * 5.5
                py_r = np.sin(rand_angle + increment_angle) * 5.5
                # check whether the initial px and py collides with any human
                collision = self.check_collision_group((px_r, py_r), self.robot.radius)
                # if the robot goal does not fall into any human groups, the goal is okay, otherwise keep generating the goal
                if not collision:
                    #print('initial pos angle:', rand_angle+increment_angle)
                    break
                increment_angle = increment_angle + 0.2

            increment_angle = increment_angle + np.pi # start at opposite side of the circle
            while True:
                gx = np.cos(rand_angle + increment_angle) * 5.5
                gy = np.sin(rand_angle + increment_angle) * 5.5
                # check whether the goal is inside the human groups
                # check whether the initial px and py collides with any human
                collision = self.check_collision_group_goal((gx, gy), self.robot.radius)
                # if the robot goal does not fall into any human groups, the goal is okay, otherwise keep generating the goal
                if not collision:
                    # print('goal pos angle:', rand_angle + increment_angle)
                    break
                increment_angle = increment_angle + 0.2

            self.robot.set(px_r, py_r, gx, gy, 0, 0, np.pi / 2)

        # for FoV environment
        else:
            if self.robot.kinematics == 'unicycle':
                angle = np.random.uniform(0, np.pi * 2)
                px = self.circle_radius * np.cos(angle)
                py = self.circle_radius * np.sin(angle)
                while True:
                    gx, gy = np.random.uniform(-self.circle_radius, self.circle_radius, 2)
                    if np.linalg.norm([px - gx, py - gy]) >= 6:  # 1 was 6
                        break
                self.robot.set(px, py, gx, gy, 0, 0, np.random.uniform(0, 2*np.pi)) # randomize init orientation

            # randomize starting position and goal position
            else:
                angle = np.random.uniform(0, np.pi * 2)
                px = self.circle_radius * np.cos(angle)
                py = self.circle_radius * np.sin(angle)
                self.robot.set(px, py, -px, -py, 0, 0, np.pi/2)


            # generate humans
            self.generate_random_human_position(human_num=(self.human_num - self.static_human_num))
            self.generate_static_human_position(human_num=self.static_human_num)


    # reset function
    def reset(self, phase='train', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """

        if self.phase is not None:
            phase = self.phase
        if self.test_case is not None:
            test_case = self.test_case

        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case  # test case is passed in to calculate specific seed to generate case

        self.global_time = 0
        self.accumulate_reward = 0
        self.success_times = 0
        self.desiredVelocity = [0.0, 0.0]
        self.humans = []
        self.goal_list = []

        self.robot_traj = [[]]
        self.human_traj = [[] for _ in range(self.human_num)]
        # train, val, and test phase should start with different seed.
        # case capacity: the maximum number for train(max possible int -2000), val(1000), and test(1000)
        # val start from seed=0, test start from seed=case_capacity['val']=1000
        # train start from self.case_capacity['val'] + self.case_capacity['test']=2000
        counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                          'val': 0, 'test': self.case_capacity['val']}

        # here we use a counter to calculate seed. The seed=counter_offset + case_counter
        np.random.seed(counter_offset[phase] + self.case_counter[phase] + self.thisSeed)
        self.generate_robot_humans(phase)

        # If configured to randomize human policies, do so
        if self.random_policy_changing:
            self.randomize_human_policies()

        # case size is used to make sure that the case_counter is always between 0 and case_size[phase]
        self.case_counter[phase] = (self.case_counter[phase] + int(1 * self.nenv)) % self.case_size[phase]

        ob = self.generate_ob(reset=True)

        # initialize potential
        self.potential = -abs(
            np.linalg.norm(np.array([self.robot.px, self.robot.py]) - np.array([self.robot.gx, self.robot.gy])))

        return ob


    # Update the humans' end goals in the environment
    # Produces valid end goals for each human
    def update_human_goals_randomly(self):
        # Update humans' goals randomly
        for human in self.humans:
            if human.isObstacle or human.v_pref == 0:
                continue
            if np.random.random() <= self.goal_change_chance:
                if not self.group_human: # to improve the runtime
                    humans_copy = []
                    for h in self.humans:
                        if h != human:
                            humans_copy.append(h)


                # Produce valid goal for human in case of circle setting
                while True:
                    angle = np.random.random() * np.pi * 2
                    # add some noise to simulate all the possible cases robot could meet with human
                    v_pref = 1.0 if human.v_pref == 0 else human.v_pref
                    gx_noise = (np.random.random() - 0.5) * v_pref
                    gy_noise = (np.random.random() - 0.5) * v_pref
                    gx = self.circle_radius * np.cos(angle) + gx_noise
                    gy = self.circle_radius * np.sin(angle) + gy_noise
                    collide = False

                    if self.group_human:
                        collide = self.check_collision_group((gx, gy), human.radius)
                    else:
                        for agent in [self.robot] + humans_copy:
                            min_dist = human.radius + agent.radius + self.discomfort_dist
                            if norm((gx - agent.px, gy - agent.py)) < min_dist or \
                                    norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                                collide = True
                                break
                    if not collide:
                        break

                # Give human new goal
                human.gx = gx
                human.gy = gy
        return

    # Update the specified human's end goals in the environment randomly
    def update_human_goal(self, human):
        # Update human's goals randomly
        if np.random.random() <= self.end_goal_change_chance:
            if not self.group_human:
                humans_copy = []
                for h in self.humans:
                    if h != human:
                        humans_copy.append(h)

            # Update human's radius now that it's reached goal
            if self.random_radii:
                human.radius += np.random.uniform(-0.1, 0.1)

            # Update human's v_pref now that it's reached goal
            if self.random_v_pref:
                human.v_pref += np.random.uniform(-0.1, 0.1)

            while True:
                angle = np.random.random() * np.pi * 2
                # add some noise to simulate all the possible cases robot could meet with human
                v_pref = 1.0 if human.v_pref == 0 else human.v_pref
                gx_noise = (np.random.random() - 0.5) * v_pref
                gy_noise = (np.random.random() - 0.5) * v_pref
                gx = self.circle_radius * np.cos(angle) + gx_noise
                gy = self.circle_radius * np.sin(angle) + gy_noise
                collide = False
                if self.group_human:
                    collide = self.check_collision_group((gx, gy), human.radius)
                else:
                    for agent in [self.robot] + humans_copy:
                        min_dist = human.radius + agent.radius + self.discomfort_dist
                        if norm((gx - agent.px, gy - agent.py)) < min_dist or \
                                norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                            collide = True
                            break
                if not collide:
                    break

            # Give human new goal
            human.gx = gx
            human.gy = gy
        return

    # Caculate whether agent2 is in agent1's FOV
    # Not the same as whether agent1 is in agent2's FOV!!!!
    # arguments:
    # state1, state2: can be agent instance OR state instance
    # robot1: is True if state1 is robot, else is False
    # return value:
    # return True if state2 is visible to state1, else return False
    def detect_visible(self, state1, state2, robot1 = False, custom_fov=None):
        if self.robot.kinematics == 'holonomic':
            real_theta = np.arctan2(state1.vy, state1.vx)
        else:
            real_theta = state1.theta
        # angle of center line of FOV of agent1
        v_fov = [np.cos(real_theta), np.sin(real_theta)]

        # angle between agent1 and agent2
        v_12 = [state2.px - state1.px, state2.py - state1.py]
        # angle between center of FOV and agent 2

        v_fov = v_fov / np.linalg.norm(v_fov)
        v_12 = v_12 / np.linalg.norm(v_12)

        offset = np.arccos(np.clip(np.dot(v_fov, v_12), a_min=-1, a_max=1))
        if custom_fov:
            fov = custom_fov
        else:
            if robot1:
                fov = self.robot_fov
            else:
                fov = self.human_fov

        if np.abs(offset) <= fov / 2:
            return True
        else:
            return False


    # for robot:
    # return only visible humans to robot and number of visible humans and visible humans' ids (0 to 4)
    def get_num_human_in_fov(self):
        human_ids = []
        humans_in_view = []
        num_humans_in_view = 0

        for i in range(self.human_num):
            visible = self.detect_visible(self.robot, self.humans[i], robot1=True)
            if visible:
                humans_in_view.append(self.humans[i])
                num_humans_in_view = num_humans_in_view + 1
                human_ids.append(True)
            else:
                human_ids.append(False)

        return humans_in_view, num_humans_in_view, human_ids


    # convert an np array with length = 34 to a JointState object
    def array_to_jointstate(self, obs_list):
        fullstate = FullState(obs_list[0], obs_list[1], obs_list[2], obs_list[3],
                              obs_list[4],
                              obs_list[5], obs_list[6], obs_list[7], obs_list[8])

        observable_states = []
        for k in range(self.human_num):
            idx = 9 + k * 5
            observable_states.append(
                ObservableState(obs_list[idx], obs_list[idx + 1], obs_list[idx + 2],
                                obs_list[idx + 3], obs_list[idx + 4]))
        state = JointState(fullstate, observable_states)
        return state


    def last_human_states_obj(self):
        '''
        convert self.last_human_states to a list of observable state objects for old algorithms to use
        '''
        humans = []
        for i in range(self.human_num):
            h = ObservableState(*self.last_human_states[i])
            humans.append(h)
        return humans


    def generate_new_robot_goal(self):
        while True:
            collide = False
            gx, gy = np.random.uniform(-(self.circle_radius + 2), self.circle_radius + 2, 2)
            if np.linalg.norm([self.robot.px - gx, self.robot.py - gy]) < 3:  # 1 was 6
                continue

            for i, agent in enumerate(self.humans):
                # keep human at least 3 meters away from robot
                if self.robot.kinematics == 'unicycle' and i == 0:
                    min_dist = self.circle_radius / 2  # Todo: if circle_radius <= 4, it will get stuck here
                else:
                    min_dist = self.robot.radius + agent.radius + self.discomfort_dist
                if norm((gx - agent.px, gy - agent.py)) < min_dist or \
                        norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        return gx, gy

    # find R(s, a), done or not, and episode information
    def calc_reward(self, action):
        # collision detection
        dmin = float('inf')

        danger_dists = []
        collision = False

        for i, human in enumerate(self.humans):
            dx = human.px - self.robot.px
            dy = human.py - self.robot.py
            closest_dist = (dx ** 2 + dy ** 2) ** (1 / 2) - human.radius - self.robot.radius

            if closest_dist < self.discomfort_dist:
                danger_dists.append(closest_dist)
            if closest_dist < 0:
                collision = True
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        reaching_goal = norm(np.array(self.robot.get_position()) - np.array(self.robot.get_goal_position())) < self.robot.radius

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            episode_info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            episode_info = Collision()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            episode_info = ReachGoal()
        elif dmin < self.discomfort_dist:
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            episode_info = Danger(dmin)
        else:
            potential_cur = np.linalg.norm(
                np.array([self.robot.px, self.robot.py]) - np.array(self.robot.get_goal_position()))
            reward = 2 * (-abs(potential_cur) - self.potential)
            self.potential = -abs(potential_cur)

            done = False
            episode_info = Nothing()

        # if the robot is near collision/arrival, it should be able to turn a large angle
        if self.robot.kinematics == 'unicycle':
            # add a rotational penalty
            # if action.r is w, factor = -0.02 if w in [-1.5, 1.5], factor = -0.045 if w in [-1, 1];
            # if action.r is delta theta, factor = -2 if r in [-0.15, 0.15], factor = -4.5 if r in [-0.1, 0.1]
            r_spin = -2 * action.r**2

            # add a penalty for going backwards
            if action.v < 0:
                r_back = -2 * abs(action.v)
            else:
                r_back = 0.
            # print(reward, r_spin, r_back)
            reward = reward + r_spin + r_back
        self.accumulate_reward += reward
        return reward, done, episode_info

    def generate_ob(self, reset):
        ob = {}

        # nodes
        visible_humans, num_visibles, human_visibility = self.get_num_human_in_fov()

        self.update_last_human_states(human_visibility, reset=reset)

        ob['robot_node'] = [self.robot.get_full_state_list_noV()]
        # edges
        # temporal edge: robot's velocity
        ob['temporal_edges'] = np.array([self.robot.vx, self.robot.vy]).reshape(1, -1)
        # spatial edges: the vector pointing from the robot position to each human's position
        spatial_edges = np.zeros((self.human_num, 2))

        for i in range(self.human_num):
            relative_pos = np.array(
                [self.last_human_states[i, 0] - self.robot.px, self.last_human_states[i, 1] - self.robot.py])
            spatial_edges[i] = relative_pos
        ob['spatial_edges'] = spatial_edges
        ob['flatten_obs'] = np.concatenate((spatial_edges.reshape(-1, ), self.robot.get_full_state_list()), axis=-1)

        goal_relative_pos = np.array([self.robot.gx - self.robot.px, self.robot.gy - self.robot.py, 1, 0]).reshape(1, -1)
        one_hot = np.array([0, 1] * self.human_num).reshape(self.human_num, -1)
        spatial_edges = np.concatenate((spatial_edges, one_hot), axis=-1)
        spatial_edges = np.concatenate((spatial_edges, goal_relative_pos), axis=0)
        dis = np.linalg.norm(spatial_edges, axis=-1)
        visible_mask = (dis < self.visible_dis).astype(int)
        visible_mask[-1] = 1

        ob['spatial_edges_transformer'] = spatial_edges
        ob['visible_masks'] = visible_mask
        ob['robot_pos'] = np.array([self.robot.px, self.robot.py, 1, 0]).reshape(1, -1)

        return ob

    # get the actions for all humans at the current timestep
    def get_human_actions(self):
        # step all humans
        human_actions = []  # a list of all humans' actions
        for i, human in enumerate(self.humans):
            # observation for humans is always coordinates
            ob = []
            for other_human in self.humans:
                if other_human != human:
                    # Chance for one human to be blind to some other humans
                    if self.random_unobservability and i == 0:
                        if np.random.random() <= self.unobservable_chance or not self.detect_visible(human,
                                                                                                     other_human):
                            ob.append(self.dummy_human.get_observable_state())
                        else:
                            ob.append(other_human.get_observable_state())
                    # Else detectable humans are always observable to each other
                    elif self.detect_visible(human, other_human):
                        ob.append(other_human.get_observable_state())
                    else:
                        ob.append(self.dummy_human.get_observable_state())

            if self.robot.visible:
                # Chance for one human to be blind to robot
                if self.random_unobservability and i == 0:
                    if np.random.random() <= self.unobservable_chance or not self.detect_visible(human, self.robot):
                        ob += [self.dummy_robot.get_observable_state()]
                    else:
                        ob += [self.robot.get_observable_state()]
                # Else human will always see visible robots
                elif self.detect_visible(human, self.robot):
                    ob += [self.robot.get_observable_state()]
                else:
                    ob += [self.dummy_robot.get_observable_state()]

            human_actions.append(human.act(ob))
        return human_actions


    def step(self, action, update=True):

        action = self.robot.policy.clip_action(action, self.robot.v_pref)
        human_actions = self.get_human_actions()

        reward, done, episode_info = self.calc_reward(action)

        # apply action and update all agents
        self.robot.step(action)
        for i, human_action in enumerate(human_actions):
            self.humans[i].step(human_action)
        self.global_time += self.time_step  # max episode length = time_limit / time_step

        # compute the observation
        ob = self.generate_ob(reset=False)

        info = {'info': episode_info}
        if done:
            info['episode_reward'] = self.accumulate_reward

        # Update all humans' goals randomly midway through episode
        if self.random_goal_changing:
            if self.global_time % 5 == 0:
                self.update_human_goals_randomly()

        # Update a specific human's goal once its reached its original goal
        if self.end_goal_changing:
            for human in self.humans:
                if norm((human.gx - human.px, human.gy - human.py)) < human.radius:
                    self.update_human_goal(human)

        return ob, [reward], [done], info


    def render(self, mode='rgb_array'):
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        from matplotlib import patches
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        robot_color = 'yellow'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        def calcFOVLineEndPoint(ang, point, extendFactor):
            FOVLineRot = np.array([[np.cos(ang), -np.sin(ang), 0],
                                   [np.sin(ang), np.cos(ang), 0],
                                   [0, 0, 1]])
            point.extend([1])
            newPoint = np.matmul(FOVLineRot, np.reshape(point, [3, 1]))
            newPoint = [extendFactor * newPoint[0, 0], extendFactor * newPoint[1, 0], 1]
            return newPoint

        fig = Figure(figsize=(10, 10), dpi=100)
        ax = fig.add_subplot()
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        artists = []
        ax.text(0, 9, 'Time: ' + str(self.global_time) + ' s', color='black', fontsize=16)

        robotX, robotY = self.robot.get_position()
        self.robot_traj[0].append([robotX, robotY])

        goal = mlines.Line2D([self.robot.gx], [self.robot.gy], color=goal_color, marker='*',
                         linestyle='None', markersize=15, label='Goal')
        ax.add_artist(goal)
        artists.append(goal)

        robot = plt.Circle((robotX, robotY), self.robot.radius, fill=True, color=robot_color)
        ax.add_artist(robot)
        artists.append(robot)

        radius = self.robot.radius
        arrowStartEnd = []

        robot_theta = self.robot.theta

        arrowStartEnd.append(
            ((robotX, robotY), (robotX + radius * np.cos(robot_theta), robotY + radius * np.sin(robot_theta))))

        for k, human in enumerate(self.humans):
            self.human_traj[k].append([human.px, human.py])
            if human.v_pref != 0:
                theta = np.arctan2(human.vy, human.vx)
                arrowStartEnd.append(
                ((human.px, human.py), (human.px + radius * np.cos(theta), human.py + radius * np.sin(theta))))

        arrows = [patches.FancyArrowPatch(*arrow, color=arrow_color, arrowstyle=arrow_style)
                  for arrow in arrowStartEnd]
        for arrow in arrows:
            ax.add_artist(arrow)
            artists.append(arrow)

        if self.robot_fov < np.pi * 2:
            FOVAng = self.robot_fov / 2
            FOVLine1 = mlines.Line2D([0, 0], [0, 0], linestyle='--')
            FOVLine2 = mlines.Line2D([0, 0], [0, 0], linestyle='--')

            startPointX = robotX
            startPointY = robotY
            endPointX = robotX + radius * np.cos(robot_theta)
            endPointY = robotY + radius * np.sin(robot_theta)

            FOVEndPoint1 = calcFOVLineEndPoint(FOVAng, [endPointX - startPointX, endPointY - startPointY],
                                               20. / self.robot.radius)
            FOVLine1.set_xdata(np.array([startPointX, startPointX + FOVEndPoint1[0]]))
            FOVLine1.set_ydata(np.array([startPointY, startPointY + FOVEndPoint1[1]]))
            FOVEndPoint2 = calcFOVLineEndPoint(-FOVAng, [endPointX - startPointX, endPointY - startPointY],
                                               20. / self.robot.radius)
            FOVLine2.set_xdata(np.array([startPointX, startPointX + FOVEndPoint2[0]]))
            FOVLine2.set_ydata(np.array([startPointY, startPointY + FOVEndPoint2[1]]))

            ax.add_artist(FOVLine1)
            ax.add_artist(FOVLine2)
            artists.append(FOVLine1)
            artists.append(FOVLine2)


        for i in range(len(self.humans)):
            if self.humans[i].v_pref != 0:
                circle = plt.Circle(self.humans[i].get_position(), self.humans[i].radius, fill=False)
            else:
                circle = plt.Circle(self.humans[i].get_position(), self.humans[i].radius, fill=True)
            ax.add_artist(circle)
            artists.append(circle)

            circle.set_color(c='r')
            if self.detect_visible(self.robot, self.humans[i], robot1=True):
                circle.set_color(c='g')
            ax.text(self.humans[i].px - 0.1, self.humans[i].py + 0.4, str(i), color='black', fontsize=12)

        r = mlines.Line2D([100], [100], color='yellow', marker='.', linestyle='None',
                          markersize=25, label='Robot')
        g = mlines.Line2D([100], [100], color='red', marker='*', linestyle='None',
                          markersize=15, label='Goal')
        h = mlines.Line2D([100], [100], color='black', marker='o', linestyle='None',
                          markersize=15, markerfacecolor='white', label='Human')

        ax.legend([r, g, h], ['Robot', 'Goal', 'Human'], fontsize=16)

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()
        img = np.frombuffer(s, np.uint8).reshape((height, width, 4))

        plt.close('all')

        return img

    def draw(self):
        robot_traj = np.array(self.robot_traj)
        human_traj = np.array(self.human_traj)
        robot_goal = np.array([[self.robot.gx, self.robot.gy]])

        nav_time = self.global_time + self.time_step

        return robot_traj, robot_goal, human_traj, nav_time

