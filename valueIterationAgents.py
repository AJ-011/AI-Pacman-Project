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
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
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
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        # Iterate for the specified number of times
        for _ in range(self.iterations):

            # A temp buffer for new values
            newVal = util.Counter()

            # Iterate through all MDP states
            for state in self.mdp.getStates():

                # If statement to skip terminal state
                if self.mdp.isTerminal(state) == False:

                    # Initialize max Q value as smallest possible value
                    maxQ = float('-inf')

                    # Iterate though all possible actions
                    for action in self.mdp.getPossibleActions(state):

                        # Compute Q-value
                        qval = self.computeQValueFromValues(state, action)  
                        # Keep track of the best Q-value
                        maxQ = max(maxQ, qval)  

                    # Check if Q value was found
                    if maxQ != float('-inf'):  

                        # Store the best Q-value for this state
                        newVal[state] = maxQ  

            # Apply updates after all states are processed  
            self.values = newVal   




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
        
        # Initialize q val as 0
        qVal = 0

        # Iterate through all the possible next states
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state,action):
            
            # Find reward
            reward = self.mdp.getReward(state,action,nextState)
            # Bellman eq to get qVal
            qVal += prob * (reward + (self.discount)*(self.values[nextState]))

        return qVal




    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
               
        # Initialize best action
        bestAction = None

        # Return None if in terminal state
        if self.mdp.isTerminal(state):
            return bestAction
        
        # Initialze maxQ as -inf
        maxQ = float('-inf')

        # Iterate through all the actions
        for action in self.mdp.getPossibleActions(state):
            
            # Get q val
            qVal = self.computeQValueFromValues(state,action)
            
            # If computed qVal is higher than maxQ
            if qVal > maxQ:

                # Update maxQ and bestAction
                maxQ = qVal
                bestAction = action

        return bestAction


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # Get all states
        states = self.mdp.getStates()
        # Counter to track state index
        stateIndex = 0

        # Iterate for the specified range
        for _ in range(self.iterations):
            
            # Get current state
            state = states[stateIndex]

            # Check if we are at terminal state
            if self.mdp.isTerminal(state) == False:

                # Initialize maxQ as -inf
                maxQ = float('-inf')

                # Iterate through all possible actions
                for action in self.mdp.getPossibleActions(state):
                    
                    # Compute qVal
                    qVal = self.computeQValueFromValues(state,action)
                    # Update maxVal if needed
                    maxQ = max(maxQ, qVal)

                # Check if a maxQ was found
                if maxQ != float('-inf'):
                   
                    # Update Q value
                    self.values[state] = maxQ

            # Increment state index
            stateIndex += 1

            # Reset state index if we are at the end
            if stateIndex >= len(states):
                stateIndex = 0



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # Initialize dictionary to store state predecessors
        predecessors = {}

        # Initialize empty set for each state
        for state in self.mdp.getStates():
            predecessors[state] = set()

        # Iterate all states to get predecessors
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                
                # Get all possisble states and probabilities
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):

                    # if probability is > 0 for next state add its predecessor
                    if prob > 0:
                        predecessors[nextState].add(state)

        # Initialize priority queue
        priority_queue = util.PriorityQueue()

        # Compute initial priorities and populate the queue
        for state in self.mdp.getStates():

            # Skip terminal states
            if self.mdp.isTerminal(state) == False:  

                # Get highest Q-value for state
                maxQ = self.getMaxQValue(state)  
                diff = abs(self.values[state] - maxQ)

                # Use negative diff to prioritize larger changes
                priority_queue.push(state, -diff)  

        # Process priority queue for specified iterations
        for _ in range(self.iterations):

            if priority_queue.isEmpty():

                # Stop if no more states to process
                break  

            # Get the most important state
            state = priority_queue.pop() 

            if not self.mdp.isTerminal(state):

                # Update the state's value
                self.values[state] = self.getMaxQValue(state)

            # Update predecessors and push them if needed
            for predecessor in predecessors[state]:

                if not self.mdp.isTerminal(predecessor):

                    maxQ = self.getMaxQValue(predecessor)
                    diff = abs(self.values[predecessor] - maxQ)

                    if diff > self.theta:
                        
                        # Push if the change is significant
                        priority_queue.update(predecessor, -diff)  

    def getMaxQValue(self, state):
        """
        Returns the highest Q-value among all actions in the given state.
        If no actions are available, returns 0.
        """
        actions = self.mdp.getPossibleActions(state)

        if not actions:
            return 0  # No actions available (for terminal states)

        return max(self.computeQValueFromValues(state, action) for action in actions)



