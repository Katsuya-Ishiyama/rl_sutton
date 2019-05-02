# -*- coding: utf-8 -*-

import logging
from pathlib import Path
import re
from typing import List
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def calculate_average_by_play_count(data: List[float]):
    data_array = np.array(data)
    _average_by_play_count = data_array.mean(axis=0)
    return _average_by_play_count


def extract_exploratory_rate(dir: Path):
    logger.debug('source text: {}'.format(dir.name))
    exploratory_rate_pattern = re.compile(r'^exploratory(0\.*\d*)$')
    _extract_result = exploratory_rate_pattern.findall(dir.name)
    logger.debug('extract result: {}'.format(_extract_result))
    exploratory_rate = float(_extract_result[0])
    return exploratory_rate


def calculate_average_rewards(output_dir):
    _output_dir = Path(output_dir)
    exploratory_rate = extract_exploratory_rate(_output_dir)
    logger.debug('Exploratory Rate: {}'.format(exploratory_rate))

    play_count_list = []
    rewards_list = []
    for file in _output_dir.iterdir():
        logger.debug('reading {}'.format(file))
        src_data = pd.read_csv(file)
        if not play_count_list:
            play_count_list.extend(src_data.play_count.tolist())
            logger.debug('Extracted Play Counts: {}'.format(play_count_list))
        rewards_list.append(src_data.reward.tolist())

    average_rewards_by_play_counts = pd.DataFrame(
        data={
            'play_count': play_count_list,
            'exploratory_rate': exploratory_rate,
            'average_reward': calculate_average_by_play_count(rewards_list)
        }
    )
    logger.debug('average_rewards_by_play_counts: {}'.format(average_rewards_by_play_counts.head()))

    return average_rewards_by_play_counts


def calculate_average_suitable_action_rate(output_dir):
    _output_dir = Path(output_dir)
    exploratory_rate = extract_exploratory_rate(_output_dir)
    logger.debug('Exploratory Rate: {}'.format(exploratory_rate))

    play_count_list = []
    suitable_action_rate_list = []
    for file in _output_dir.iterdir():
        src_data = pd.read_csv(file)
        if not play_count_list:
            play_count_list.extend(src_data.play_count.tolist())
            logger.debug('Extracted Play Counts: {}'.format(play_count_list))
        is_suitable_action = src_data.action == src_data.most_suitable_action
        suitable_action_count = is_suitable_action.cumsum()
        suitable_action_rate = suitable_action_count.div(src_data.play_count)
        suitable_action_rate_list.append(suitable_action_rate.tolist())

    average_suitable_action_rate_by_play_count = pd.DataFrame(
        data={
            'play_count': play_count_list,
            'exploratory_rate': exploratory_rate,
            'average_suitable_action_rate': calculate_average_by_play_count(suitable_action_rate_list)
        }
    )
    logger.debug('calculate_average_suitable_action_rate: {}'.format(average_suitable_action_rate_by_play_count.head()))

    return average_suitable_action_rate_by_play_count
