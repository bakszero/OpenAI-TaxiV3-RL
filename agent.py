import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, epsilon=1.,alpha=0.2, gamma=1.,episode=1):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.episode = episode
        
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        
    
        if state in self.Q:
            prob = [1 - self.epsilon + self.epsilon*1./self.nA if (x==np.argmax(self.Q[state])) else self.epsilon*1./self.nA for x in range(self.nA)]
            return np.random.choice(np.arange(self.nA), p = prob)
        else:
            
            return np.random.choice(np.arange(self.nA))

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        
        self.epsilon = max(1./self.episode,0.001)
        if done:
            self.episode+=1
            self.Q[state][action] = self.Q[state][action] + self.alpha*(reward - self.Q[state][action])
        else:
            next_action = self.select_action(next_state)
            #policy = np.ones(self.nA)*self.epsilon/self.nA
            #policy[np.argmax(self.Q[next_state])] = 1 - self.epsilon + self.epsilon*1./self.nA
            
            #  3 choices:
            # 1. Expected Sarsa
            # 2. SarsaMax (Q-Learning)
            # 3. Sarsa(0)
            #expected_val = np.sum(self.Q[next_state]*policy)
            expected_val = np.max(self.Q[next_state])
            #expected_val = self.Q[next_state][next_action]
            self.Q[state][action] += self.alpha*(reward+ (self.gamma*expected_val)-self.Q[state][action])
