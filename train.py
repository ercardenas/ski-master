import sys

import gym
import time
import numpy as np
import cv2
from gym.envs import registration
from gym.utils import play
from algorithm import ski_learning
from algorithm import q_learning

##
## Different model configurations to train.
##

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

def ramNNLearner():
    env = gym.make('Skiing-ram-v0')
    # Build learner
    actions_func = ski_learning.get_actions_for_env(env)
    
    
    learner = q_learning.NNQLearningAlgorithm(
        actions=actions_func,
        discount=0.9,
        model=ski_learning.ski_ram_nn_model(),
        explorationProb=0.3,
        learningRate=0.05)
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

    nn_model = ski_learning.ski_image_nn_model_2()
    print(nn_model.summary())
    learner = q_learning.NNQLearningAlgorithm(
        actions=actions_func,
        discount=0.1,
        explorationProb=0.3,
        model=nn_model,
        learningRate=1)
    return env, learner

def imageNNFlowLearner():
    env = FlowEnv(gym.make('Skiing-v0'), verbose=True)
    # Build learner 
    actions_func = ski_learning.get_actions_for_env(env)

    nn_model = ski_learning.ski_image_nn_model_flow()
    print(nn_model.summary())
    learner = q_learning.NNQLearningAlgorithm(
        actions=actions_func,
        discount=0.8,
        explorationProb=0.3,
        model=nn_model,
        learningRate=1)
    return env, learner

##
## Expand the type of environments available.
##

class FlowEnv(object):
    """ Environment that expands an image envorinment with motion flow. """
    
    def __init__(self, env, verbose=False):
        self.env = env
        self.previousState = np.zeros((250,160,3))
        self.verbose = verbose

    def reset(self):
        observation = self.env.reset()
        out = self.combineImages(observation, observation)
        self.previousState = observation
        return out
        
    def step(self, action):
        new_observation, reward, game_over, info = self.env.step(action)
        merged = self.combineImages(self.previousState, new_observation)
        self.previousState = new_observation
        return merged, reward, game_over, info

    def combineImages(self, previousState, currentState):
        pb, pg, pr = cv2.split(previousState)
        cb, cg, cr = cv2.split(currentState)

        shape = cb.shape + (2,)
        b = np.zeros(shape, dtype=cb.dtype)
        g = np.zeros(shape, dtype=cb.dtype)
        r = np.zeros(shape, dtype=cb.dtype)
        
        b = cv2.calcOpticalFlowFarneback(pb, cb, b, 0.5, 1, 5, 5, 5, 1.1, 0)
        g = cv2.calcOpticalFlowFarneback(pg, cg, g, 0.5, 1, 5, 5, 5, 1.1, 0)
        r = cv2.calcOpticalFlowFarneback(pr, cr, r, 0.5, 1, 5, 5, 5, 1.1, 0)

        b1, b2 = cv2.split(b)
        g1, g2 = cv2.split(g)
        r1, r2 = cv2.split(r)
        
        merged = cv2.merge((b1,g1,r1,
                            b2,g2,r2,
                            cb.astype('float32')/255,
                            cg.astype('float32')/255,
                            cr.astype('float32')/255))
        
        if self.verbose:
            cv2.imshow("ax1", merged[:,:,:3])
            cv2.imshow("ax2", merged[:,:,3:6])
            cv2.imshow("img", merged[:,:,6:])
            cv2.waitKey(1)
        return merged
        
    def __getattr__(self, name):
        return getattr(self.env, name)


##
## Main program
##
    
def main(argv):

    env, learner = imageNNFlowLearner()
    # env, learner = ramNNLearner()
    print(env.get_action_meanings())
    
    for i in range(10000000):
        print("")
        print("iteration: {}".format(i))
        verbose = (i+1 % 10) == 0

        # Train on a fully exploration mode, and then once on using
        # the actual learned policy
        if ((i+1)%2) == 0:
            print("Exploit")
            learner.explorationProb = 0.0
        else:
            print("Explore")
            learner.explorationProb = 1.0

        records = trainOnce(env, learner, True)

        # Reverse the list to propagate the information from the last
        # step to the first one.
        records.reverse()
        trainReplay(learner, records, 3)
        
        if True:
            print("Start Learned weigths")
            print(learner.weights)
            print("Finish Learned weigths")


##
## Helper functions for the training.
##
            
# sarsa state, action reward, newstate
def trainOnce(env, learner, verbose):

    records = []
    
    observation = env.reset()
    game_over = False

    all_rewards = []
    while not game_over:

        if verbose:
            env.render()
        action = learner.getAction(observation)
        
        new_observation, reward, game_over, info = env.step(action)
        all_rewards.append(reward)
        # Only learn in the final state
        if game_over:
            reward = (sum(all_rewards) /1000) + 40
        else:
            reward = 0
             
        records.append((observation, action, reward, new_observation))
        if len(records) > 3000:
            records.pop(0)
        learner.incorporateFeedback(observation, action, reward, new_observation, verbose)
        observation = new_observation
        if verbose:
            time.sleep(0.0001)
    
    print("sum_rewards : {}".format(sum(all_rewards)))
    print("sum_rewards_post_processing : {}".format(reward))
    
    return records

def trainReplay(learner, records, n):
    for i in range(n):
        for r in records:
            observation, action, reward, new_observation = r
            learner.incorporateFeedback(observation, action, reward, new_observation, True)

            
if __name__ == '__main__':
    main(sys.argv)
