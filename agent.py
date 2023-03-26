from collections import defaultdict
import random
from numpy import argmax

from policy import LinearPolicy
from learner import ReinforceAlgo

random.seed(0)

class Agent:
    
    def __init__(self, initial_state, action_size) -> None:
        # self.all_actions = [i for i in range(5)]
        self.policy = LinearPolicy(len(initial_state), action_size)
        self.learner = ReinforceAlgo(self.policy)
        
        
    def choose_action(self, cur_state):
        # return argmax(self.state_value[cur_state]) if random.random() > self.epsilon\
        #     else random.randint(0, len(self.all_actions)-1)
        return self.policy.get_action(cur_state)
    
    def add_sars(self, old_state, action, reward, new_state):
        # self.sars_pairs.append([old_state, action, reward])
        self.learner.add_sars(old_state, action, reward, new_state)
    
    def end_episode(self, print_state_trace: bool, train_policy: bool):
        if print_state_trace:
            for i, (state, action, reward) in enumerate(self.learner.sars_pairs):
                print(f"{i} - state {state}, action {action}, reward {reward}")
        if not train_policy: return 0
        ret_val = self.learner.update_policy(self.policy)
        return ret_val
            