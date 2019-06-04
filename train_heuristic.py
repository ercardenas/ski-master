import random
import sys
import cv2

import numpy as np
import gym


NUM_GAMES = 1
THETA_DIFF_THRESHOLD = 0.015

# Constants for debug use
VERBOSE = False
FRAME_DISPLAY_DURATION = 10

# Speed up color matching by removing the bottom part of the frame
CROP_OBSERVATION = True

# Color in BGR order representing the objects on the Atari screen
PLAYER_COLOR = [214, 92, 92]
FLAGES_RED_COLOR = [184, 50, 50]
FLAGES_BLUE_COLOR = [66, 72, 200]


def imshow(observation):
  cv2.imshow("Heuristic Agent", observation)
  cv2.waitKey(FRAME_DISPLAY_DURATION)


def getIndexesForColor(observation, color):
    if CROP_OBSERVATION:
        observation = observation[:200] # 0 is the top of the frame
    return np.where(np.sum(observation==color, -1)==3)


def getPlayerPosition(observation):
    """ Returns row, col pair representing the player's center position.

    observation: a frame of the game given by the Atari environment
    """
    indexes = getIndexesForColor(observation, PLAYER_COLOR)
    return indexes[0].mean(), indexes[1].mean()


def getFlagsPosition(observation):
    """ Returns row, col pair representing the flags' center position.

    Flags can be in either red or blue color, so we would attempt to extract
    whichever exists.

    observation: a frame of the game given by the Atari environment
    """
    indexes = getIndexesForColor(observation, FLAGES_RED_COLOR)
    if np.sum(indexes) == 0:
        indexes = getIndexesForColor(observation, FLAGES_BLUE_COLOR)
    return indexes[0].mean(), indexes[1].mean()


def getTheta(observation):
    """ Returns the angel between skier and flags' vertical/horizontal distance.

    observation: a frame of the game given by the Atari environment
    """
    player_pos = getPlayerPosition(observation)
    flags_pos = getFlagsPosition(observation)

    row_delta = flags_pos[0] - player_pos[0]
    col_delta = flags_pos[1] - player_pos[1]
    theta = np.arctan2(row_delta, col_delta)

    if VERBOSE:
        print("skier position: {}".format(player_pos))
        print("flags position: {}".format(flags_pos))
        print("theta: ", theta)
    return theta


def getAction(theta, prev_theta):
    if theta - prev_theta > THETA_DIFF_THRESHOLD:
        return 2
    elif theta - prev_theta < -THETA_DIFF_THRESHOLD:
        return 1
    else:
        return 0


def trainOnce(env):
    observation = env.reset()
    iterations = 0
    game_over = False
    all_rewards = []

    prev_theta = getTheta(observation)

    while not game_over:
        theta = getTheta(observation)
        action = getAction(theta, prev_theta)
        observation, reward, game_over, info = env.step(action)
        all_rewards.append(reward)
        prev_theta = theta

        if iterations % 10 == 0:
            imshow(observation)
        iterations += 1
    return sum(all_rewards)


def main(argv):
    env_name = "Skiing-v0"
    env = gym.make(env_name)

    for i in range(NUM_GAMES):
        final_rewards = trainOnce(env)
        print("rewards in game {}: {}".format(i + 1, final_rewards))


if __name__ == '__main__':
    main(sys.argv)
