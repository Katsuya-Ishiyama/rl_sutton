# -*- coding: utf-8 -*-

import csv
from datetime import datetime
import logging
import os
from typing import List, Dict
from tqdm import tqdm
import numpy as np

logger = logging.getLogger()


class NArmedBanditEnvironment(object):

    def __init__(self, arm: int):
        self.arm = arm
        self.arm_list = list(range(1, arm+1))
        self.true_action_values = None
        self.most_suitable_action_index = None

    def initialize(self):
        _this_class_name = __class__.__name__
        logger.info("Initializing {}...".format(_this_class_name))
        self._create_true_action_values()
        self._calculate_most_suitable_action_index()
        logger.info("{} has been initialized.".format(_this_class_name))

    def _calculate_most_suitable_action_index(self):
        _action_index_list = list(self.true_action_values.keys())
        _action_value_list = list(self.true_action_values.values())
        _max_action_value_list_index = np.argmax(_action_value_list)
        self.most_suitable_action_index = _action_index_list[_max_action_value_list_index]
        logger.info('Most suitable action: {}'.format(self.most_suitable_action_index))

    def _create_true_action_values(self):
        _true_action_values_list = np.random.normal(0, 1, self.arm).tolist()
        self.true_action_values = {i: v for i, v in zip(self.arm_list, _true_action_values_list)}
        logger.info("True action values: {}".format(self.true_action_values))

    def create_action_values(self) -> List[float]:
        if self.true_action_values is None:
            raise Exception("Create true action values before you run this method.")
        _true_action_values = list(self.true_action_values.values())
        _action_values_list = np.random.normal(_true_action_values, 1).tolist()
        _action_values = {i: v for i, v in zip(self.arm_list, _action_values_list)}
        logger.debug("Created action values: {}".format(_action_values))
        return _action_values

    def run(self, arm: int):
        _selected_arm = arm
        _action_values = self.create_action_values()
        reward: float = _action_values[arm]
        logger.debug("Selected arm index: {}, Reward: {}".format(_selected_arm, reward))
        return reward


class BanditLogger(object):

    def __init__(self):
        self._logs = []
        self.most_suitable_action = None

    def register(self, play_count, arm, reward, exploratory_rate):

        if self.most_suitable_action is None:
            logger.error('most_suitable_action is not set.')
            raise Exception('most_suitable_action is not set.')

        _log = {
            "play_count": play_count,
            "action": arm,
            "reward": reward,
            "exploratory_rate": exploratory_rate,
            "most_suitable_action": self.most_suitable_action
        }
        self._logs.append(_log)
        log_msg = 'Registered log. (Play count: {}, Action: {}, Reward: {}, Exploratory rate: {})'
        logger.debug(log_msg.format(play_count, arm, reward, exploratory_rate))

    def write_logs_to_csv(self, path):
        with open(path, 'w') as f:
            fieldnames = ['play_count', 'action', 'reward', 'most_suitable_action', 'exploratory_rate']
            csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(self._logs)

    def get_rewards_by_action(self) -> Dict[int, List[float]]:
        _rewards_by_action: Dict[int, List[float]] = {}
        for _log in self._logs:
            _action: int = _log["action"]
            _reward: float = _log["reward"]
            _rewards_by_action.setdefault(_action, [])
            _rewards_by_action[_action].append(_reward)
        return _rewards_by_action

    def get_rewards_by_play_count(self) -> List[float]:
        _rewards_by_play_count = []
        for _log in self._logs:
            _rewards_by_play_count.append(_log["reward"])
        return _rewards_by_play_count


class NArmedBanditAgent(object):

    def __init__(self, environment):
        self.environment = environment
        self.arm_list = environment.arm_list
        self.play_count = 0
        self.play_log = BanditLogger()
        self.play_log.most_suitable_action = environment.most_suitable_action_index
        self.exploratory_rate = None

    def select_policy(self, eps=0.1):
        if (eps < 0) or (1 < eps):
            logger.error('eps cannot be handled.')
            raise ValueError('eps cannot be handled.')
        if eps == 0:
            logger.warning('The greedy policy will be selected at this time.')
        if eps == 1:
            logger.warning('The exploratory policy will be selected at this time.')
        is_exploratory = (np.random.uniform(0, 1) < eps) or (self.play_count == 1)
        if is_exploratory:
            _selected_action_index = self.select_policy_exploratory()
        else:
            _selected_action_index = self.select_policy_greedy()
        logger.info('Selected action: {}'.format(_selected_action_index))
        return _selected_action_index

    def select_policy_greedy(self) -> int:
        _estimated_action_values = self.estimate_action_values()
        _estimated_action_values_list = list(_estimated_action_values.values())
        _estimated_action_values_list_index = np.argmax(_estimated_action_values_list)
        _selected_action_value_index = self.arm_list[_estimated_action_values_list_index]
        logger.info('A greedy policy has been selected.')
        return _selected_action_value_index

    def select_policy_exploratory(self) -> int:
        _selected_action_value_index = np.random.choice(self.arm_list)
        logger.info('A exploratory policy has been selected.')
        return _selected_action_value_index

    def receive_reward(self, arm: int, reward: float):
        self.play_log.register(self.play_count, arm, reward, self.exploratory_rate)

    def estimate_action_values(self) -> Dict[int, float]:
        rewards = self.play_log.get_rewards_by_action()
        _estimated_action_values = {i: 0 for i in self.arm_list}
        for _arm, rewards_list in rewards.items():
            _estimated_action_values[_arm] = np.mean(rewards_list)
        logger.info('Estimated action values: {}'.format(_estimated_action_values))
        return _estimated_action_values

    def play(self, exploratory_rate=0.1):
        self.play_count += 1
        logger.info('Play counts: {}'.format(self.play_count))
        self.exploratory_rate = exploratory_rate
        logger.info('Exploratory rate: {}'.format(self.exploratory_rate))
        selected_action = self.select_policy(self.exploratory_rate)
        reward = self.environment.run(selected_action)
        logger.info('Reward: {}'.format(reward))
        self.receive_reward(selected_action, reward)

    def write_logs_to_csv(self, path):
        logger.info('Exporting logs... Path: {}'.format(path))
        self.play_log.write_logs_to_csv(path)
        logger.info('Exporting logs has done. Path: {}'.format(path))


def simulate_n_armed_bandit(arm, exploratory_rate, play, iterations):

    bandit_env = NArmedBanditEnvironment(arm=arm)

    for simulation_num in tqdm(range(1, iterations+1)):
        logger.info('--------------------------------------------------')
        logger.info('Start simulation No.{}'.format(simulation_num))
        logger.info('arm: {}, exploratory_rate: {}, play: {}, iterations: {}'.format(arm, exploratory_rate, play, iterations))
        bandit_env.initialize()
        agent = NArmedBanditAgent(bandit_env)
        for _ in range(play):
            agent.play(exploratory_rate)
        filename = 'n_armed_bandit_arm{}_exploratory{}_simulation{}.csv'.format(arm, exploratory_rate, simulation_num)
        save_path = os.path.join('output', 'exploratory{}'.format(exploratory_rate), filename)
        agent.write_logs_to_csv(path=save_path)


def main():
    ARM_NUM = 10
    MAX_PLAY_COUNT = 1000
    SIMULATION_COUNT = 2000

    simulate_n_armed_bandit(
        arm=ARM_NUM,
        exploratory_rate=0,
        play=MAX_PLAY_COUNT,
        iterations=SIMULATION_COUNT
    )

    simulate_n_armed_bandit(
        arm=ARM_NUM,
        exploratory_rate=0.01,
        play=MAX_PLAY_COUNT,
        iterations=SIMULATION_COUNT
    )

    simulate_n_armed_bandit(
        arm=ARM_NUM,
        exploratory_rate=0.1,
        play=MAX_PLAY_COUNT,
        iterations=SIMULATION_COUNT
    )


if __name__ == '__main__':
    current_time_str = datetime.now().strftime('%Y%m%d%H%M%S')
    file_handler = logging.FileHandler('bandit_simulation_{}.log'.format(current_time_str))
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[file_handler]
    )

    main()
