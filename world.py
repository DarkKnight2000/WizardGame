from agent import Agent
import pygame
# pygame.init()
import gym
import time

import random
random.seed(0)


class World:
    def __init__(self) -> None:
        # self.grid_size = 5
        # self.grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        # self.grid[0][0] = 10
        # self.grid[1][1] = -100
        
        self.ep_length = 0
        self.map_size = 500
        self.world_draw = WorldDraw()
        self.PLAYER_SIZE = 5 # half size
        self.PLAYER_SPEED = 2
        
        self.agent_pos: list[float] = None
        self.target_pos: list[float] = None
        self.reset_agent()
        self.reset_target()

        self.agent = Agent(self._encode_state(), 5)
        self.actions = [0,0,0,0,0]
        
    def reset_agent(self):
        self.actions = [0,0,0,0,0]
        self.agent_pos = [self.map_size/2, self.map_size/2]
        
    def reset_target(self):
        if not self.agent_pos: self.reset_agent()
        self.target_pos = [self.map_size * random.random(), self.map_size * random.random()]
        while self.is_target_hit():
            self.target_pos = [self.map_size * random.random(), self.map_size * random.random()]
            self.target_pos[0] = min(max(self.target_pos[0], self.PLAYER_SIZE), self.map_size-self.PLAYER_SIZE)
            self.target_pos[1] = min(max(self.target_pos[1], self.PLAYER_SIZE), self.map_size-self.PLAYER_SIZE)
        self.ep_length = 0
        
    def is_target_hit(self):
        if abs(self.target_pos[0] - self.agent_pos[0]) < 2 * self.PLAYER_SIZE:
            if abs(self.target_pos[1] - self.agent_pos[1]) < 2 * self.PLAYER_SIZE:
                return True
        return False
        
    # return if episode ended in this update step
    def update(self) -> int:
        self.ep_length += 1
        old_state = self._encode_state()
        action = self.agent.choose_action(old_state)
        
        # take action and calculate reward
        reward = -0.01
        # if(action[0] < 0.5):
        if(action == 1):
            if self.PLAYER_SIZE > self.agent_pos[0]: reward = -1
            if self.target_pos[0] < self.agent_pos[0]: reward = 1
            self.agent_pos[0] = max(self.PLAYER_SIZE, self.agent_pos[0]-self.PLAYER_SPEED)
            self.actions[0] += 1
        # else:
        elif(action == 2):
            if self.map_size-self.PLAYER_SIZE < self.agent_pos[0]: reward = -1
            if self.target_pos[0] > self.agent_pos[0]: reward = 1
            self.agent_pos[0] = min(self.map_size-self.PLAYER_SIZE, self.agent_pos[0]+self.PLAYER_SPEED)
            self.actions[1] += 1
        # if(action[1] < 0.5):
        if(action == 3):
            if self.PLAYER_SIZE > self.agent_pos[1]: reward = -1
            if self.target_pos[1] < self.agent_pos[1]: reward = 1
            self.agent_pos[1] = max(self.PLAYER_SIZE, self.agent_pos[1]-self.PLAYER_SPEED)
            self.actions[2] += 1
        # else:
        elif(action == 4):
            if self.map_size-self.PLAYER_SIZE < self.agent_pos[1]: reward = -1
            if self.target_pos[1] > self.agent_pos[1]: reward = 1
            self.agent_pos[1] = min(self.map_size-self.PLAYER_SIZE, self.agent_pos[1]+self.PLAYER_SPEED)
            self.actions[3] += 1
        # elif action != 0:
        #     print(f"Invalid action {action}")
        
        reached_target = self.is_target_hit()
        if reached_target:
            reward = 100
            print("reached target")
        
        self.agent.add_sars(old_state, action, reward, self._encode_state())
        
        # select return for the update step        
        if reached_target: return 1
        if self.ep_length > 2 * self.map_size: return -1
        else: return 0
        
    def run_episode(self, print_state_trace: bool, draw_game: bool, train_policy: bool, is_soft_reset: bool) -> tuple:
        self.reset_agent()
        if not is_soft_reset: self.reset_target()
        
        update_val = self.world_draw.draw_episode(self, draw_game)
            
        total_reward = self.agent.end_episode(print_state_trace, train_policy)
        return update_val, self.ep_length, total_reward
    
    def _encode_state(self):
        return [self.agent_pos[0]/self.map_size, self.agent_pos[1]/self.map_size, self.target_pos[0]/self.map_size, self.target_pos[1]/self.map_size]
        

class WorldDraw:
    
    def __init__(self) -> None:
        self.fonts = {}
    
    def draw_text(self, screen: pygame.Surface, text:str, font_size:int, pos, anchor: int = 1):
        if font_size not in self.fonts:
            fontname = pygame.font.get_default_font()
            sysfont = pygame.font.Font(fontname, font_size)
            self.fonts[font_size] = sysfont
          
        render_text = self.fonts[font_size].render(text, True, (0,0,0))
        textRect: pygame.Rect = render_text.get_rect()
        if anchor == 0: textRect.center = pos
        elif anchor == 1: textRect.topleft = pos
        screen.blit(render_text, textRect)
            
    def draw_episode(self, world: World, shld_draw: bool) -> int:
        
        show_time = 0
        last_world_update = 0
        after_finish_time = 1000 if shld_draw else 0
        
        if(shld_draw): pygame.init()

        # Set up the drawing window
        BORDER = 50
        if(shld_draw): screen = pygame.display.set_mode([world.map_size + 2*BORDER, world.map_size + 2*BORDER])
        gameclock = pygame.time.Clock()     

        # Run until the user asks to quit
        running = True
        while running:
            
            if shld_draw: dt = gameclock.tick(60)
            else: dt = 1000 / 60

            if shld_draw:
                # Did the user click the window close button?
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        running = False
                    
            # Update only if last update was not the end
            if last_world_update == 0:
                last_world_update = world.update()
                show_time += dt
                # if(show_time >= 2 * world.map_size * 1000 / 60): last_world_update = -1
            else:
                # print('after game')
                after_finish_time -= dt
                show_time += dt
                if after_finish_time <= 0: running = False
                # print(f'{show_time} {after_finish_time}')
                           

            if(shld_draw): 
                # Fill the background with white
                screen.fill((255, 255, 255))
                
                self.draw_text(screen, f"ep time {(show_time/1000)}", 10, (20, 20))
                self.draw_text(screen, f"last update {last_world_update}", 10, (20, 30))
                self.draw_text(screen, f"ep len {world.ep_length}", 10, (20, 40))
                self.draw_text(screen, f"agent pos {world.agent_pos}", 10, (20, 50))

                # draw world
                pygame.draw.rect(
                    screen,
                    (0,0,0),
                    pygame.Rect(BORDER-10, BORDER-10, world.map_size+20, world.map_size+20), width=10)
                
                pygame.draw.rect(
                    screen,
                    (100,100,100),
                    pygame.Rect(
                        world.agent_pos[0]-world.PLAYER_SIZE+BORDER,
                        world.agent_pos[1]-world.PLAYER_SIZE+BORDER,
                        world.PLAYER_SIZE*2, world.PLAYER_SIZE*2), width=2)
                
                pygame.draw.rect(
                    screen,
                    (200,200,100),
                    pygame.Rect(
                        world.target_pos[0]-world.PLAYER_SIZE+BORDER,
                        world.target_pos[1]-world.PLAYER_SIZE+BORDER,
                        world.PLAYER_SIZE*2, world.PLAYER_SIZE*2))

                # Flip the display
                pygame.display.flip()

        # Done! Time to quit.
        if(shld_draw):
            pygame.quit()
            self.fonts.clear()
        
        return last_world_update
        
      
if __name__ == "__main__":
    
    world_type = "mine"
    
    if world_type == "mine":
        world = World()
        
        # ep_id = 0
        # prev_ep_ret = (-1, 100, -1)
        # while prev_ep_ret[2] < world.grid[0][0] - world.grid_size*2 + 2 and ep_id < 500:
        #     print(f"Running episode {ep_id}...", end="")
        #     prev_ep_ret = world.run_episode(ep_id % 100 == 0)
        #     print("done!")
            
        #     ep_id += 1
            
        # print(f"epsilon {world.agent.epsilon}")
        # for i, (state, _, reward) in enumerate(world.agent.old_sars_pairs):
        #     print(f"{i} - {state}, {reward}")
            
            
        # wrld_draw = WorldDraw()
        # wrld_draw.draw(500)
        
        ep_id = 1
        prev_ep_ret = (-1, 100, [-1,-1])
        while ep_id < 1000:
            print(f"Running episode {ep_id}...", end="")
            prev_ep_ret = world.run_episode(ep_id % 500 == 0, ep_id % 500 == 0, True, False)
            print(f"done! - reward {prev_ep_ret[2]}, ep len {prev_ep_ret[1]}")
            print(world.actions)
            ep_id += 1
            
        print(world.agent.policy.model)
        world.run_episode(False, True, False, True)
        
    else:
        # env = gym.make(world_type, render_mode="rgb_array")

        # state_size = env.observation_space.shape[0]
        # action_size = env.action_space.n
        # print(action_size)
        
        # agent = Agent(env.observation_space.sample(), action_size)
        
        # state, info = env.reset(seed=42)
        # ep_id = 1
        # step = 0
        # while ep_id <= 10:
        #     action = agent.choose_action(state)
        #     new_state, reward, terminated, truncated, info = env.step(action.item())
        #     agent.add_sars(state, action, reward, new_state)
        #     state = new_state
        #     step += 1
            
        #     env.render()
        #     time.sleep(0.01)

        #     if terminated or truncated or step > 1000:
        #         agent.end_episode(False, True)
        #         observation, info = env.reset()
        #         ep_id += 1
        #         step = 0
        #         print(f"ep {ep_id-1} done")
        
        # env.close()
        
        
        env = gym.make("CartPole-v1", render_mode="human")
        observation, info = env.reset(seed=123, options={})
        
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        print(action_size)
        
        agent = Agent(observation, action_size)

        ep_id = 0
        while ep_id <= 100:
            done = False
            observation, info = env.reset()
            if ep_id % 10 == 0: print('episode')
            while not done:
                action = agent.choose_action(observation)
                old_state = observation
                observation, reward, terminated, truncated, info = env.step(action.item())
                if ep_id % 10 == 0: env.render()
                
                done = terminated or truncated
                agent.add_sars(old_state, action, reward, observation)
            agent.end_episode(False, True)
            ep_id += 1

        env.close()