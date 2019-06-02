import math
import random
import collections
import cv2

import numpy as np


# Abstract class: an RLAlgorithm performs reinforcement learning.  All it needs
# to know is the set of available actions to take.  The simulator (see
# simulate()) will call getAction() to get an action, perform the action, and
# then provide feedback (via incorporateFeedback()) to the RL algorithm, so it can adjust
# its parameters.
class RLAlgorithm(object):
    # Your algorithm will be asked to produce an action given a state.
    def getAction(self, state): raise NotImplementedError("Override me")

    # We will call this function when simulating an MDP, and you should update
    # parameters.
    # If |state| is a terminal state, this function will be called with (s, a,
    # 0, None). When this function is called, it indicates that taking action
    # |action| in state |state| resulted in reward |reward| and a transition to state
    # |newState|.
    def incorporateFeedback(self, state, action, reward, newState): raise NotImplementedError("Override me")


# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(RLAlgorithm):
    def __init__(self, actions, discount, explorationProb=0.2, learningRate=1.0):
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb
        self.weights = collections.defaultdict(float)
        self.numIters = 0
        self.learningRate = learningRate

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        raise Exception("getQ Is Not Implemented")
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState, verbose=False):
        raise Exception("incorporateFeedback is not implemented")


# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class LinearQLearningAlgorithm(QLearningAlgorithm):
    
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2, learningRate=1.0):
        super(LinearQLearningAlgorithm, self).__init__(actions=actions,
                                                       discount=discount,
                                                       explorationProb=explorationProb,
                                                       learningRate=learningRate)
        self.featureExtractor = featureExtractor
        
    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score


    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState, verbose=False):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)

        if newState is None:
            return
        
        # Update weights to
        stepSize = self.getStepSize()
        # print('step size', stepSize)
        VnewState = self.discount * max([self.getQ(newState, newAction) for newAction in self.actions(newState)])
        target = reward + VnewState
        
        VcurrentState = self.getQ(state, action)
        residual = VcurrentState - target 

        if verbose:
            print()
            print()
            print("VcurrentState", VcurrentState)
            print("VnewState", VnewState)
            print("reward", reward)
            print("residual", residual)
            print("target", target)
            print("stepSize", stepSize)
        
        for f, v in self.featureExtractor(state, action):
            self.weights[f] -= (stepSize * (VcurrentState - target) * v) * self.learningRate
            if v > 0.0:
                print(((VcurrentState - target) * v))
        # END_YOUR_CODE



# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# model: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class NNQLearningAlgorithm(QLearningAlgorithm):
    
    def __init__(self, actions, discount, model, explorationProb=0.2, learningRate=1.0):
        super(NNQLearningAlgorithm, self).__init__(actions=actions,
                                                       discount=discount,
                                                       explorationProb=explorationProb,
                                                       learningRate=learningRate)
        # Model should receive as inputs state=state(w x h x c, float), action=action(1x1,int)
        self.model = model
    
    # Return the Q function associated with the weights and features
    def getQ(self, state, action):

        state_array = np.array([state])
        action_array = np.zeros((1,3))
        action_array[0][action] = 1.0
        
        score = self.model.predict([state_array, action_array])
        # print(score.shape)
        return score[0]


    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState, verbose=False):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)

        if newState is None:
            return
        
        # Update weights to
        stepSize = self.getStepSize()
        # print('step size', stepSize)
        VnewStates = [self.getQ(newState, newAction) for newAction in self.actions(newState)]
        VnewState = self.discount * max(VnewStates)
        target = reward + VnewState
        
        VcurrentState = self.getQ(state, action)

        if verbose:
            verb_img = np.zeros((600, 1000))
            cv2.putText(verb_img,
                        "scores: " + str(VnewStates),
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255))
            cv2.putText(verb_img,
                        "VcurrentState: " + str(VcurrentState),
                        (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255))
            cv2.putText(verb_img,
                        "reward: " + str(reward),
                        (50, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255))
            cv2.putText(verb_img,
                        "target: " + str(target),
                        (50, 140),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255))
            cv2.imshow("debug", verb_img)
            cv2.waitKey(1)
            
            # print()
            # print()
            # print("VcurrentState", VcurrentState)
            # print("VnewState", VnewState)
            # print("reward", reward)
            # print("target", target)
            # print("stepSize", stepSize)
        

        # To train the NN we need, (state, action), target
        state_array = np.array([state])
        action_array = np.zeros((1,3))
        action_array[0][action] = 1.0
        sample_weight_array = np.array([stepSize * self.learningRate])
        target_array = np.array(target)
        self.model.train_on_batch([state_array, action_array], target_array, sample_weight_array)
        
        # END_YOUR_CODE
        
        
