import numpy as np
import random
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

class LinearPolicy:
    
    def __init__(self, feature_len, actions_len) -> None:
        # self.weights = np.random.random((actions_count, feature_len))
        self.model = torch.nn.Sequential(
            torch.nn.Linear(feature_len, actions_len),
            torch.nn.ReLU(),
            torch.nn.Linear(actions_len, actions_len),
            torch.nn.Softmax(dim=0)
        )
        # print(self.weights)
    
    def get_action(self, state):
        probs = self.model(torch.Tensor(state))
        # print(self.model(torch.Tensor(state)))
        # print(probs)
        try:
            action = torch.multinomial(probs, 1)
            # print(action)
        except RuntimeError:
            action = 0
            print('error', probs)
        # print(action)
        return action
        
        # return self.model(torch.Tensor(state))
    
    # def get_weights_len(self):
    #     return len(self.weights)
    
    # def log_grad(self):
    #     return 1 / self.weights
    
    # def update_weights(self, action, delta):
    #     self.weights[action] += delta