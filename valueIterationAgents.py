# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        # length of the iterations
        for len in range(iterations):
            # util.Counter that holds the value and hash it for the current state
            hashValues = util.Counter()

            # the current states
            for state in self.mdp.getStates():
                # optimal value in each of the iterations
                opt_value = float('-inf')
                # initialize the opt_value as negative because from now on each of the value will be greater than it

                # now, iterate the actions
                for action in mdp.getPossibleActions(state):
                    # initialize the value as 0
                    v = 0
                    # get probable transition from the current state to the next one
                    for t in self.mdp.getTransitionStatesAndProbs(state, action):
                        # follow the equation that the slides shown for calculating the V value
                        v += t[1] * (self.mdp.getReward(state, action, t[0]) + discount * self.values[t[0]])

                    # now, compare the calculated v to the optimal value
                    # if it is larger, then the optimal value will be updated
                    if opt_value < v:
                        opt_value = v

                # now update the util.Counter hashed values to optimal value
                if opt_value != float('-inf'):
                    hashValues[state] = opt_value

            # From now, all the V values are calculated and this needs to be shown in the agent value list
            for s in self.mdp.getStates():
                self.values[s] = hashValues[s]

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        # take a look at each action and state in the current position
        total_value = []
        nextState_prob_list = self.mdp.getTransitionStatesAndProbs(state, action)
        for (nextState, prob) in nextState_prob_list:
            reward = self.mdp.getReward(state, action, nextState)
            total_value.append(prob * (reward + self.discount * self.values[nextState]))
        return sum(total_value)

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # form up the same iterations as the __init__ function
        opt_value = float('-inf')
        opt_action = ''
        action_list = self.mdp.getPossibleActions(state)

        # get all possible actions
        for action in action_list:
            # get the q value
            q_v = self.computeQValueFromValues(state, action)

            # compare and update the opt_value
            if opt_value < q_v:
                opt_value = q_v
                opt_action = action

        # if none of the action is optimal, then just return None
        if opt_action == '':
            return None
        else:
            return opt_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
