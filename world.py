from agent import Agent

class World:
    def __init__(self) -> None:
        self.grid_size = 5
        self.grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.grid[0][0] = 10
        self.grid[1][1] = -100
        
        self.ep_length = 0

        self.agent = Agent()
        self.reset()
        
    def reset(self):
        self.agent_pos = [self.grid_size-1, self.grid_size-1]
        self.ep_length = 0
        
    # return if episode ended in this update step
    def update(self) -> int:
        self.ep_length += 1
        action = self.agent.choose_action(self.agent_pos)
        
        # take action and calculate reward
        old_state = self.agent_pos.copy()
        
        if(action == 1):
            self.agent_pos[0] = max(0, self.agent_pos[0]-1)
        if(action == 2):
            self.agent_pos[0] = min(self.grid_size-1, self.agent_pos[0]+1)
        if(action == 3):
            self.agent_pos[1] = max(0, self.agent_pos[1]-1)
        if(action == 4):
            self.agent_pos[1] = min(self.grid_size-1, self.agent_pos[1]+1)
        reward = self.grid[self.agent_pos[0]][self.agent_pos[1]] - 1
        
        self.agent.add_sars(old_state, action, reward, self.agent_pos)
        
        # select return for the update step        
        if self.agent_pos == [0,0]: return 1
        if self.ep_length > 100: return -1
        else: return 0
        
    def run_episode(self, print_state_trace: bool) -> tuple:
        self.reset()
        
        while True:
            update_val = self.update()
            if update_val != 0:
                break
            
        total_agent_reward = self.agent.end_episode(print_state_trace)
        return update_val, self.ep_length, total_agent_reward
        
        
if __name__ == "__main__":
    
    world = World()
    
    ep_id = 0
    prev_ep_ret = (-1, 100, -1)
    while prev_ep_ret[2] < world.grid[0][0] - world.grid_size*2 + 2 and ep_id < 500:
        print(f"Running episode {ep_id}...", end="")
        prev_ep_ret = world.run_episode(ep_id % 100 == 0)
        print("done!")
        
        ep_id += 1
        
    print(f"epsilon {world.agent.epsilon}")
    for i, (state, _, reward) in enumerate(world.agent.old_sars_pairs):
        print(f"{i} - {state}, {reward}")
        