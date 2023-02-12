from collections import defaultdict
import random
from numpy import argmax

random.seed(0)

class Agent:
    
    def __init__(self) -> None:
        self.all_actions = [i for i in range(5)]
        self.state_value = defaultdict(lambda : [0 for _ in (self.all_actions)])
        self.epsilon = 1
        
        # episode specific
        self.sars_pairs = []
        
        # policy specific
        self.num_visits = defaultdict(lambda : [0 for _ in (self.all_actions)])
        
        # debug
        self.old_sars_pairs = []
        
    def choose_action(self, pos) -> int:
        cur_state = self._encode_state(pos)
        return argmax(self.state_value[cur_state]) if random.random() > self.epsilon\
            else random.randint(0, len(self.all_actions)-1)
    
    def _encode_state(self, pos):
        return (pos[0], pos[1])
    
    def add_sars(self, old_pos, action, reward, new_pos):
        self.sars_pairs.append([self._encode_state(old_pos), action, reward])
    
    def end_episode(self, print_state_trace: bool) -> int:
        # set total rewards for each step
        total_reward = 0
        for i, (state, action, reward) in enumerate(reversed(self.sars_pairs)):
            total_reward += reward
            self.sars_pairs[len(self.sars_pairs) - 1 - i][2] = total_reward

        # Update average rewards
        for (state, action, reward) in self.sars_pairs:
            
            cur_visits = self.num_visits[state][action]
            avg_reward = cur_visits * self.state_value[state][action]
            try:
                avg_reward += reward
                cur_visits += 1
                avg_reward /= cur_visits
            except OverflowError:
                # print(f"overflow error {reward}")
                avg_reward = -100
                cur_visits = 1

            # avoid overflow error
            if avg_reward < -100:
                avg_reward = -100
                cur_visits = 1
            self.num_visits[state][action] = cur_visits
            self.state_value[state][action] = avg_reward
            
        if self.epsilon > 0.0005: self.epsilon *= 0.99
        
        if print_state_trace:
            print(f"epsilon {self.epsilon}")
            for i, (state, _, reward) in enumerate(self.sars_pairs):
                print(f"{i} - {state}, {reward}")
        
        self.old_sars_pairs.clear()
        self.old_sars_pairs = self.sars_pairs.copy()
        self.sars_pairs.clear()
        
        # print(self.state_value)
        return total_reward
            