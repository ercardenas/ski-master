import sys

import gym
import time
from gym.envs import registration
from gym.utils import play
from algorithm import ski_learning
from algorithm import q_learning

# env = gym.make('CartPole-v0')


def ramLearner():
    env = gym.make('Skiing-ram-v0')
    # Build learner 
    actions_func = ski_learning.get_actions_for_env(env)
    learner = q_learning.LinearQLearningAlgorithm(
        actions=actions_func,
        discount=0.5,
        featureExtractor=ski_learning.ski_ram_base_feature_extractor,
        explorationProb=0.3,
        learningRate=0.00005)
    return env, learner

def imageLearner():
    env = gym.make('Skiing-v0')
    # Build learner 
    actions_func = ski_learning.get_actions_for_env(env)
    learner = q_learning.LinearQLearningAlgorithm(
        actions=actions_func,
        discount=0.5,
        featureExtractor=ski_learning.ski_image_resized_action_feature_extractor,
        # featureExtractor=ski_learning.ski_image_resized_feature_extractor,
        # featureExtractor=ski_learning.ski_image_base_feature_extractor,
        explorationProb=0.3,
        learningRate=100)
    return env, learner

def imageNNLearner():
    env = gym.make('Skiing-v0')
    # Build learner 
    actions_func = ski_learning.get_actions_for_env(env)

    nn_model = ski_learning.ski_image_nn_model()
    print(nn_model.summary())
    learner = q_learning.NNQLearningAlgorithm(
        actions=actions_func,
        discount=0.9,
        explorationProb=0.3,
        model=nn_model,
        learningRate=1)
    return env, learner

def main(argv):

    env, learner = imageNNLearner()
    
    for i in range(10000000):
        print("iteration: {}".format(i))
        verbose = (i % 1) == 0

        # Train on a fully exploration mode, and then once on using
        # the actual learned policy
        if ((i+1)%5) == 0:
            print("Exploit")
            learner.explorationProb = 0.1
        else:
            print("Explore")
            learner.explorationProb = 0.9
            
        
        trainOnce(env, learner, True)
        
        if True:
            print("Start Learned weigths")
            print(learner.weights)
            print("Finish Learned weigths")
    

def trainOnce(env, learner, verbose):

    observation = env.reset()
    game_over = False

    all_rewards = []
    while not game_over:

        if verbose:
            env.render()
        action = learner.getAction(observation)
        
        new_observation, reward, game_over, info = env.step(action)
        learner.incorporateFeedback(observation, action, reward, new_observation, True)
        observation = new_observation

        all_rewards.append(reward)
        # print("Start Learned weigths Inter")
        # print(learner.weights)
        # print("Finish Learned weigths Inter")

        
        # # Print and wait
        # print(
        #     "ob:{ob}, reward: {reward}, game_over: {game_over}, d: {d}".format(
        #         ob=observation, reward=reward, game_over=game_over, d=info
        #     ))
        if verbose:
            time.sleep(0.0001)
    print("sum_rewards : {}".format(sum(all_rewards)))


if __name__ == '__main__':
    main(sys.argv)
