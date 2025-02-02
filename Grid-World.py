# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 22:44:46 2021

@author: Kieffer
"""

import random
import matplotlib.pyplot as plt

# Set up the initial environment
num_rows = 4
num_cols = 4

actions = [(1, 0), (-1, 0), (0, 1), (0, -1)] # Right, Left, Up, Down
num_actions = len(actions)

states = [(i,j) for i in range(num_cols) for j in range(num_rows)]
initial_state = (0,0)
final_state = (3,0)
holes = [(2,0),(2,1)]

# Reward evaluation function
def reward(old_state,action):
    if old_state == final_state:
        return 0

    # Apply action
    candidate_state = (old_state[0]+action[0],old_state[1]+action[1])
    
    # checks whether action is valid
    if candidate_state in states:
        if candidate_state in holes:
            return -10
        else:
            return -0.1
    else:
        return -1


# Transition probability evaluation
def transition_prob(old_state,action,new_state):
    # Case of final state
    if old_state == final_state:
        if new_state == final_state:
            return 1
        else:
            return 0
    
    # Apply action
    candidate_state = (old_state[0]+action[0],old_state[1]+action[1])
    
    # checks whether action is valid
    if candidate_state in states:
        if candidate_state == new_state:
            return 1
    else:
        if old_state == new_state:
            return 1
    return 0


# Policy evaluation algorithm for deterministic policy
def policy_eval(policy,gamma,theta=1e-3):
    """
    Inputs : policy : policy to evaluate
             gamma : discount factor
             theta : accuracy criterion
    """    
    # Initialization
    V = [0 for state in states]
    Delta = float('inf') # in order to make sure that my loop will run once at least
    k = 0
    
    while Delta > theta:
        Delta = 0
        k=k+1

        for s in range(0,len(states)):
            # Back-up of V[s]
            v = V[s]
            
            V[s] = 0

            # Policy evaluation with deterministic policy
            for sp in range(0,len(states)):
                V[s] += transition_prob(states[s],policy[s],states[sp])\
                    *(reward(states[s],policy[s]) + gamma * V[sp])
            
            Delta = max(Delta,abs(v-V[s]))

    print("Nb iterations : ", k)        
    return V


# Policy iteration algorithm
def policy_iteration(policy,gamma,theta=1e-3):
    """
    Inputs : policy : policy to evaluate
             gamma : dicount factor
             theta : accuracy criterion
    """    
    policy_stable = False

    # Loop until policy is stable    
    while policy_stable == False:
        policy_stable = True
        
        # Policy evaluation
        V = policy_eval(policy,gamma,theta)
            
        # Policy improvement step
        for s in range(0,len(states)): # here s is the index of state not the state itself since it is easier to deal with integer not tuple where stet is pair of coord
            old_action = policy[s]
            q = []
            for action in actions:
                tmp = 0
                for sp in range(0,len(states)):
                    tmp += transition_prob(states[s],action,states[sp])\
                        *(reward(states[s],policy[s]) + gamma * V[sp])
                q.append(tmp) # we are evaluating the action value funct for that state
            
            idx = q.index(max(q))
            policy[s] = actions[idx]
            
            if old_action != policy[s]:
                policy_stable = False

        V = policy_eval(policy,gamma,theta)
        display_value_policy(V,policy)
                
    return policy
                
# Value evaluation algorithm for deterministic policy
def value_iteration(gamma,theta=1e-3):
    """
    Inputs : gamma : dicount factor
             theta : accuracy criterion
    Outputs : V : optimal value function
              policy : associated policy
    """    


                
# Displays the Value function
def display_value_policy(V,policy):
    
    plt.xlim([-0.5, num_cols-0.5])
    plt.ylim([-0.5, num_rows-0.5])
    
    for s in range(0,len(states)):
        state = states[s]
        pi = policy[s]
        plt.text(state[0], state[1], "{:10.3f}".format(V[s]), size=12,
                 ha="center", va="center")
        
        plt.arrow(state[0], state[1], pi[0]/3, pi[1]/3, head_width=0.05, head_length=0.1, fc='r', ec='r')

    plt.show()
    
# Initial policy
#   defines initial policy
policy = [random.choice(actions) for state in states]
print("Initial policy")

for i in range(len(states)):
    print(states[i]," : ",policy[i])

#   evaluates initial policy 
gamma = 0.9
V1 = policy_eval(policy,gamma)
gamma = 0.99
V2 = policy_eval(policy,gamma)
gamma = 0.999
V3 = policy_eval(policy,gamma)
    
#   display
display_value_policy(V1,policy)
display_value_policy(V2,policy)
display_value_policy(V3,policy)


# Policy iteration algorithm
gamma = 0.9
optimal_policy = policy_iteration(policy,gamma)

#   evaluates optimal policy

V_optimal = policy_eval(optimal_policy,gamma)


#   display
display_value_policy(V_optimal,optimal_policy)

# Value iteration algorithm
def value_iteration(gamma, theta=1e-3):
    """
    Inputs:
    - gamma: discount factor
    - theta: accuracy criterion (default: 1e-3)
    """
    # Initialization
    V = [0 for state in states]
    Delta = float('inf')  # in order to make sure that my loop will run once at least
    k = 0

    while Delta > theta:
        Delta = 0
        k = k + 1

        for s in range(0, len(states)):
            # Back-up of V[s]
            v = V[s]

            V[s] = 0

            # Policy evaluation with deterministic policy
            for sp in range(0, len(states)):
                V[s] += transition_prob(states[s], policy[s], states[sp]) \
                        * (reward(states[s], policy[s]) + gamma * V[sp])

            Delta = max(Delta, abs(v - V[s]))

    print("Nb iterations : ", k)
    return V

    # Construct the optimal policy based on the updated value function
    optimal_policy = []
    for s in range(len(states)):
        v = V[s]
        q = []
        for action in actions:
            tmp = 0
            for sp in range(len(states)):
                tmp += transition_prob(states[s], action, states[sp]) \
                       * (reward(states[s], action) + gamma * V[sp])
            q.append(tmp)
        idx = q.index(max(q))
        optimal_policy.append(actions[idx])

    return optimal_policy

# Set the discount factor
gamma = 0.9

# Call the value_iteration function
A_optimal = value_iteration(gamma)

# Display the optimal policy
print(optimal_policy)
display_value_policy(A_optimal,optimal_policy)