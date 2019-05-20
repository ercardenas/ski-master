import sys

import gym
import time
from gym.envs import registration
from gym.utils import play
from algorithm import ski_learning
from algorithm import q_learning

# env = gym.make('CartPole-v0')



def main(argv):
    
    env = gym.make('Skiing-ram-v0')
    # Build learner 
    actions_func = ski_learning.get_actions_for_env(env)
    learner = q_learning.QLearningAlgorithm(actions_func,
                                            0.9,
                                            ski_learning.ski_ram_base_feature_extractor)
    for i in range(10):
        trainOnce(env, learner)

        # print("Start Learned weigths")
        # print(learner.weights)
        # print("Finish Learned weigths")
        

def trainOnce(env, learner):

    observation = env.reset()
    game_over = False

    while not game_over:

        env.render()
        action = learner.getAction(observation)
        
        new_observation, reward, game_over, info = env.step(action)
        learner.incorporateFeedback(observation, action, reward, new_observation)
        observation = new_observation

        # print("Start Learned weigths Inter")
        # print(learner.weights)
        # print("Finish Learned weigths Inter")

        
        # # Print and wait
        # print(
        #     "ob:{ob}, reward: {reward}, game_over: {game_over}, d: {d}".format(
        #         ob=observation, reward=reward, game_over=game_over, d=info
        #     ))
        # time.sleep(0.0001)


if __name__ == '__main__':
    main(sys.argv)
