from policy import LinearPolicy

import numpy as np
import math, sys
import torch

class ReinforceAlgo:
    
    def __init__(self, policy: LinearPolicy, gamma = 1, learn_rate = 0.01) -> None:
        
        # parameters
        self.gamma = gamma
        self.learn_rate = learn_rate
        self.optimizer = torch.optim.Adam(policy.model.parameters(), lr=self.learn_rate)
        
        # episode specific
        self.sars_pairs = []

        # debug
        self.old_sars_pairs = []
    
    def add_sars(self, state, action, reward, new_state):
        self.sars_pairs.append([state, action, reward])
        
    def update_policy(self, policy: LinearPolicy) -> int:
        
        # set total rewards for each step
        # print('\nrewards', [r for (s,a,r) in self.sars_pairs])
        total_reward = 0
        for i, (_, _, reward) in enumerate(reversed(self.sars_pairs)):
            # total_reward[0] = reward[0] + total_reward[0] * self.gamma
            # total_reward[1] = reward[1] + total_reward[1] * self.gamma
            total_reward = reward + total_reward * self.gamma
            # total_reward = torch.add(total_reward * self.gamma, torch.Tensor(reward))
            self.sars_pairs[len(self.sars_pairs) - 1 - i][2] = total_reward
            # print(reward, total_reward)
            
            # if math.isinf(total_reward[0]) or math.isinf(total_reward[1]):
            #     print("Infinite reward")
            #     sys.exit(1)
                
        # print([r for (s,a,r) in self.sars_pairs])
        rewards = torch.Tensor([r for (s,a,r) in self.sars_pairs]) 
        states = torch.Tensor(np.array([s for (s,a,r) in self.sars_pairs])) 
        actions = torch.stack([a for (s,a,r) in self.sars_pairs]) 
        # print(rewards)
        rewards /= torch.max(torch.abs(rewards)) + 1

        # gradient = np.zeros_like(policy.weights[0])
        # for (_, action, reward) in reversed(self.sars_pairs):
        #     gradient += policy.log_grad()[action] * reward
        
        pred_prob = policy.model(states)
        action_prob = torch.gather(pred_prob, dim=1, index=actions.long().view(len(self.sars_pairs),-1)).squeeze()
        # print(action_prob)
        
        # print(action_prob)
        # print(action_prob.squeeze())
        # loss = -torch.sum(torch.log(action_prob)*rewards)
        # # print('rewards', rewards)
        # print(loss)
        
        BATCH_SIZE = 64
        # batch_id = 0
        # loss = 0
        # while batch_id*BATCH_SIZE < len(self.sars_pairs):
        #     loss_sum = (torch.log(action_prob[batch_id*BATCH_SIZE:batch_id*BATCH_SIZE+BATCH_SIZE])
        #                       * rewards[batch_id*BATCH_SIZE:batch_id*BATCH_SIZE+BATCH_SIZE])
        #     # print(loss_sum)
        #     loss += -torch.sum(loss_sum)
        #     # print('rewards', rewards)
        #     # print('loss', loss)
        #     batch_id += 1
            
        loss = -torch.sum(torch.log(action_prob) * rewards)
        loss /= int(len(self.sars_pairs) / BATCH_SIZE) + 1
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # print('loss:',loss)
        
        self.old_sars_pairs.clear()
        self.old_sars_pairs = self.sars_pairs.copy()
        self.sars_pairs.clear()
        
        return total_reward
