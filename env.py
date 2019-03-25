from time import time
from datetime import datetime
import numpy as np
import random
import pygame
from pygame.locals import *
from pygame.transform import smoothscale
import warnings
import logging

from agents import *
from items import *
from settings import s, e

from random import shuffle
from collections import deque
from utils import game_state_to_img,get_reward_coins,get_reward_crates,get_reward_fight
import matplotlib.pyplot as plt
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

ACTION_TO_VALUE = {'UP':0,'DOWN':1,'LEFT':2,'RIGHT':3,'BOMB':4,'WAIT':5}
obser_fn = game_state_to_img
if s.phase == 'coins':
    reward_fn = get_reward_coins
elif s.phase in ['crates25','crates50','crates75']:
    reward_fn = get_reward_crates
elif s.phase == 'fight':
    reward_fn = get_reward_fight
else:
    raise NotImplementedError

class BaseAgent(object):
    coin_trophy = smoothscale(pygame.image.load('assets/coin.png'), (15, 15))
    suicide_trophy = smoothscale(pygame.image.load('assets/explosion_2.png'), (15, 15))
    time_trophy = pygame.image.load('assets/hourglass.png')

    def __init__(self, name, color, train_flag,record=False):
        self.game_state = None
        self.name = name
        self.color = color

        self.train_flag = train_flag
        self.shade = pygame.Surface((30, 30), SRCALPHA)
        self.shade.fill((0, 0, 0, 208))
        self.avatar = pygame.image.load(f'assets/robot_{self.color}.png')

        self.x, self.y = 1, 1
        self.total_score = 0
        self.bomb_timer = s.bomb_timer + 1
        self.explosion_timer = s.explosion_timer + 1
        self.bomb_power = s.bomb_power
        self.bomb_type = Bomb
        self.round = 0

        self.game_state = None
        self.record = record
        if self.record==True:
            self.memory = []
            self.memory_size = 0

    def act(self, game_state):
        raise NotImplementedError
    def update_record(self,game_state,done):
        raise NotImplementedError

    def reset(self, round):
        self.round = round
        self.dead = False
        self.score = 0
        self.step_events = []
        self.bombs_left = 1
        self.trophies = []
        self.game_state = None
        self.next_action = None
        if self.record:
            self.eps_observations = []
            self.eps_actions = []
            self.eps_rewards = []
            self.sample_per_ep = 10
    def update_score(self, delta):
        """Add delta to both the current round's score and the total score."""
        self.score += delta
        self.total_score += delta

    def get_state(self):
        """Provide information about this agent for the global game state."""
        return (self.x, self.y, self.name, self.bombs_left, self.score)

    def render(self, screen, x, y):
        """Draw the agent's avatar to the screen at the given coordinates."""
        screen.blit(self.avatar, (x, y))
        if self.dead:
            screen.blit(self.shade, (x, y))

    def make_bomb(self):
        """Create a new Bomb object at current agent position."""
        return self.bomb_type((self.x, self.y), self,
                              self.bomb_timer, self.bomb_power, self.color)

class SimpleAgent(BaseAgent):
    def __init__(self, name, color, train_flag,record=False):
        super(SimpleAgent, self).__init__(name, color, train_flag,record)
        """Called once before a set of games to initialize data structures etc.

            The 'self' object passed to this method will be the same in all other
            callback methods. You can assign new properties (like bomb_history below)
            here or later on and they will be persistent even across multiple games.
        """
        np.random.seed()
        # Fixed length FIFO queues to avoid repeating the same actions
        self.bomb_history = deque([], 5)
        self.coordinate_history = deque([], 20)
        # While this timer is positive, agent will not hunt/attack opponents
        self.ignore_others_timer = 0

    def discount_and_norm_rewards(self, gamma):

        # discount episode rewards
        discounted_ep_rs = np.zeros(len(self.eps_actions))
        running_add = 0
        for t in reversed(range(0, len(self.eps_rewards))):
            running_add = running_add * gamma + self.eps_rewards[t]
            discounted_ep_rs[t] = running_add

        return discounted_ep_rs

    def update_record(self,game_state,done):
        assert self.record
        if not self.dead:
            obs = obser_fn(self.game_state)
            rew = reward_fn(self.step_events,game_state)
            act = ACTION_TO_VALUE[self.next_action]
            self.eps_observations.append(obs)
            self.eps_rewards.append(rew)
            self.eps_actions.append(act)

        if done:
            # pick 10 trans per epoch and save per 1000 epoch
            discounted_eps_reward = self.discount_and_norm_rewards(gamma=0.99)
            idx = np.random.permutation(len(self.eps_actions))[:self.sample_per_ep]
            for i in idx:
                observation = self.eps_observations[i]
                action = self.eps_actions[i]
                reward = self.eps_rewards[i]
                discount_rew = discounted_eps_reward[i]
                transition = [observation, action, reward, discount_rew]
                self.memory.append(transition)
                self.memory_size += 1
                if self.memory_size % 1000 == 0:
                    print('id{} current step:{}'.format(self.name, self.memory_size))
                if self.memory_size % 20000 == 0:
                    print('save {} record'.format(self.memory_size))
                    data_matrix = np.array(self.memory)
                    np.save('./tmp/memory/{}/memory_id{}_{}.npy'.format(s.phase,self.name, self.memory_size), data_matrix)




    def act(self, game_state):
        """Called each game step to determine the agent's next action.

        You can find out about the state of the game environment via self.game_state,
        which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
        what it contains.

        Set the action you wish to perform by assigning the relevant string to
        self.next_action. You can assign to this variable multiple times during
        your computations. If this method takes longer than the time limit specified
        in settings.py, execution is interrupted by the game and the current value
        of self.next_action will be used. The default value is 'WAIT'.
        """
        if game_state['exit'] == True:
            warnings.warn('shouldn not acces')
        self.game_state = game_state
        # Gather information about the game state
        arena = self.game_state['arena']
        x, y, _, bombs_left, score = self.game_state['self']
        bombs = self.game_state['bombs']
        bomb_xys = [(x, y) for (x, y, t) in bombs]
        others = [(x, y) for (x, y, n, b, s) in self.game_state['others']]
        coins = self.game_state['coins']
        bomb_map = np.ones(arena.shape) * 5
        for xb, yb, t in bombs:
            for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
                if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                    bomb_map[i, j] = min(bomb_map[i, j], t)

        # If agent has been in the same location three times recently, it's a loop
        if self.coordinate_history.count((x, y)) > 2:
            self.ignore_others_timer = 5
        else:
            self.ignore_others_timer -= 1
        self.coordinate_history.append((x, y))

        # Check which moves make sense at all
        directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        valid_tiles, valid_actions = [], []
        for d in directions:
            if ((arena[d] == 0) and
                    (self.game_state['explosions'][d] <= 1) and
                    (bomb_map[d] > 0) and
                    (not d in others) and
                    (not d in bomb_xys)):
                valid_tiles.append(d)
        if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
        if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
        if (x, y - 1) in valid_tiles: valid_actions.append('UP')
        if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
        if (x, y) in valid_tiles: valid_actions.append('WAIT')
        # Disallow the BOMB action if agent dropped a bomb in the same spot recently
        if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')

        # Collect basic action proposals in a queue
        # Later on, the last added action that is also valid will be chosen
        action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        shuffle(action_ideas)

        # Compile a list of 'targets' the agent should head towards
        dead_ends = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 0)
                     and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
        crates = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 1)]
        targets = coins + dead_ends + crates
        # Add other agents as targets if in hunting mode or no crates/coins left
        if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
            targets.extend(others)

        # Exclude targets that are currently occupied by a bomb
        targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

        # Take a step towards the most immediately interesting target
        free_space = arena == 0
        if self.ignore_others_timer > 0:
            for o in others:
                free_space[o] = False
        d = self.look_for_targets(free_space, (x, y), targets)
        if d == (x, y - 1): action_ideas.append('UP')
        if d == (x, y + 1): action_ideas.append('DOWN')
        if d == (x - 1, y): action_ideas.append('LEFT')
        if d == (x + 1, y): action_ideas.append('RIGHT')
        if d is None:
            action_ideas.append('WAIT')

        # Add proposal to drop a bomb if at dead end
        if (x, y) in dead_ends:
            action_ideas.append('BOMB')
        # Add proposal to drop a bomb if touching an opponent
        if len(others) > 0:
            if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
                action_ideas.append('BOMB')
        # Add proposal to drop a bomb if arrived at target and touching crate
        if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
            action_ideas.append('BOMB')

        # Add proposal to run away from any nearby bomb about to blow
        for xb, yb, t in bombs:
            if (xb == x) and (abs(yb - y) < 4):
                # Run away
                if (yb > y): action_ideas.append('UP')
                if (yb < y): action_ideas.append('DOWN')
                # If possible, turn a corner
                action_ideas.append('LEFT')
                action_ideas.append('RIGHT')
            if (yb == y) and (abs(xb - x) < 4):
                # Run away
                if (xb > x): action_ideas.append('LEFT')
                if (xb < x): action_ideas.append('RIGHT')
                # If possible, turn a corner
                action_ideas.append('UP')
                action_ideas.append('DOWN')
        # Try random direction if directly on top of a bomb
        for xb, yb, t in bombs:
            if xb == x and yb == y:
                action_ideas.extend(action_ideas[:4])

        # Pick last action added to the proposals list that is also valid
        while len(action_ideas) > 0:
            a = action_ideas.pop()
            if a in valid_actions:
                self.next_action = a
                break

        # Keep track of chosen action for cycle detection
        if self.next_action == 'BOMB':
            self.bomb_history.append((x, y))

        return self.next_action

    def look_for_targets(self, free_space, start, targets):
        """Find direction of closest target that can be reached via free tiles.

        Performs a breadth-first search of the reachable free tiles until a target is encountered.
        If no target can be reached, the path that takes the agent closest to any target is chosen.

        Args:
            free_space: Boolean numpy array. True for free tiles and False for obstacles.
            start: the coordinate from which to begin the search.
            targets: list or array holding the coordinates of all target tiles.
        Returns:
            coordinate of first step towards closest target or towards tile closest to any target.
        """
        if len(targets) == 0: return None

        frontier = [start]
        parent_dict = {start: start}
        dist_so_far = {start: 0}
        best = start
        best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

        while len(frontier) > 0:
            current = frontier.pop(0)
            # Find distance from current position to all targets, track closest
            d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
            if d + dist_so_far[current] <= best_dist:
                best = current
                best_dist = d + dist_so_far[current]
            if d == 0:
                # Found path to a target's exact position, mission accomplished!
                best = current
                break
            # Add unexplored free neighboring tiles to the queue in a random order
            x, y = current
            neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
            shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in parent_dict:
                    frontier.append(neighbor)
                    parent_dict[neighbor] = current
                    dist_so_far[neighbor] = dist_so_far[current] + 1
        # Determine the first step towards the best found target tile
        current = best
        while True:
            if parent_dict[current] == start: return current
            current = parent_dict[current]

class BombeRLeWorld(object):

    def __init__(self):
        self._action_set = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
        # self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(210, 160, 3), dtype=np.float32)

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(17, 17, 1), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self._action_set))
        self.setup_logging()
        if s.gui:
            self.setup_gui()

        # Available robot colors
        self.colors = ['blue', 'green', 'yellow', 'pink']

        # n<3  n in-loop agent and 1 agent from external
        # Only the first agent is always the training agent!!! It uses the action from step(action)
        # self.agents = self.agents = [
        #         SimpleAgent(str(0), 'blue', False,True),
        #         # SimpleAgent(str(1), self.colors.pop(), False,True),
        #         # SimpleAgent(str(2), self.colors.pop(), False,True),
        #         # SimpleAgent(str(3), self.colors.pop(), False,True),
        #     ]
        if s.phase == 'fight':
            self.agents = [
                BaseAgent(str(0), 'blue', True),
                SimpleAgent(str(1), self.colors.pop(), False),
                SimpleAgent(str(2), self.colors.pop(), False),
                SimpleAgent(str(3), self.colors.pop(), False),
            ]
        else:
            self.agents=[
                        BaseAgent(str(0), 'blue', True)
                       ]

        # Get the game going
        self.round = 0
        self.running = False
        self.state = None
    def reset(self):
        # reset all for a new_round()
        self.running = True
        self.round += 1
        pygame.display.set_caption(f'BombeRLe | Round #{self.round}')
        self.current_step = 0
        self.ep_reward = 0
        self.trainer_visit = np.zeros((17,17))
        self.active_agents = []
        self.bombs = []
        self.explosions = []
        self.round_id = f'Replay {datetime.now().strftime("%Y-%m-%d %H-%M-%S")}'
        # Arena with wall and crate layout
        # 1 for crates −1 for stone walls and 0 for free tiles.
        self.arena = (np.random.rand(s.cols, s.rows) < s.crate_density).astype(int)
        self.arena[:1, :] = -1
        self.arena[-1:, :] = -1
        self.arena[:, :1] = -1
        self.arena[:, -1:] = -1
        for x in range(s.cols):
            for y in range(s.rows):
                if (x + 1) * (y + 1) % 2 == 1:
                    self.arena[x, y] = -1
        # Starting positions
        #  craters(1) -> 0 (free tiles)
        self.start_positions = [(1, 1), (1, s.rows - 2), (s.cols - 2, 1), (s.cols - 2, s.rows - 2)]
        random.shuffle(self.start_positions)
        for (x, y) in self.start_positions:
            for (xx, yy) in [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if self.arena[xx, yy] == 1:
                    self.arena[xx, yy] = 0

        # Distribute coins evenly
        self.coins = []
        for i in range(3):
            for j in range(3):
                n_crates = (self.arena[1 + 5 * i:6 + 5 * i, 1 + 5 * j:6 + 5 * j] == 1).sum()
                while True:
                    x, y = np.random.randint(1 + 5 * i, 6 + 5 * i), np.random.randint(1 + 5 * j, 6 + 5 * j)
                    if n_crates == 0 and self.arena[x, y] == 0:
                        self.coins.append(Coin((x, y)))
                        self.coins[-1].collectable = True
                        break
                    elif self.arena[x, y] == 1:
                        self.coins.append(Coin((x, y)))
                        break

        # Reset agents and distribute starting positions
        for agent in self.agents:
            agent.reset(self.round)
            self.active_agents.append(agent)
            agent.x, agent.y = self.start_positions.pop()

        obervation = self.get_observation()
        self.state = self.get_info()
        return obervation
    def step(self, input_action):
        self.current_step += 1
        for agent in self.agents:
            agent.step_events= []
        self.logger.info(f'STARTING STEP {self.current_step}')
        self.poll_and_run_agents(self._action_set[input_action])
        # Coins
        for coin in self.coins:
            if coin.collectable:
                for a in self.active_agents:
                    if a.x == coin.x and a.y == coin.y:
                        coin.collectable = False
                        self.logger.info(f'Agent <{a.name}> picked up coin at {(a.x, a.y)} and receives 1 point')
                        a.update_score(s.reward_coin)
                        a.step_events.append(e.COIN_COLLECTED)
                        a.trophies.append(Agent.coin_trophy)
        # Bombs
        for bomb in self.bombs:
            # Explode when timer is finished
            if bomb.timer <= 0:
                self.logger.info(f'Agent <{bomb.owner.name}>\'s bomb at {(bomb.x, bomb.y)} explodes')
                bomb.owner.step_events.append(e.BOMB_EXPLODED)
                blast_coords = bomb.get_blast_coords(self.arena)
                # Clear crates
                for (x, y) in blast_coords:
                    if self.arena[x, y] == 1:
                        self.arena[x, y] = 0
                        bomb.owner.step_events.append(e.CRATE_DESTROYED)
                        # Maybe reveal a coin
                        for c in self.coins:
                            if (c.x, c.y) == (x, y):
                                c.collectable = True
                                self.logger.info(f'Coin found at {(x,y)}')
                                bomb.owner.step_events.append(e.COIN_FOUND)
                # Create explosion
                screen_coords = [(s.grid_offset[0] + s.grid_size * x, s.grid_offset[1] + s.grid_size * y) for (x, y) in
                                 blast_coords]
                self.explosions.append(Explosion(blast_coords, screen_coords, bomb.owner))
                bomb.active = False
                bomb.owner.bombs_left += 1
            # Progress countdown
            else:
                bomb.timer -= 1
        self.bombs = [b for b in self.bombs if b.active]
        # Explosions
        agents_hit = set()
        for explosion in self.explosions:
            # Kill agents
            if explosion.timer > 1:
                for a in self.active_agents:
                    if (not a.dead) and (a.x, a.y) in explosion.blast_coords:
                        agents_hit.add(a)
                        # Note who killed whom, adjust scores
                        if a is explosion.owner:
                            self.logger.info(f'Agent <{a.name}> blown up by own bomb')
                            a.step_events.append(e.KILLED_SELF)
                            explosion.owner.trophies.append(Agent.suicide_trophy)
                        else:
                            self.logger.info(f'Agent <{a.name}> blown up by agent <{explosion.owner.name}>\'s bomb')
                            self.logger.info(f'Agent <{explosion.owner.name}> receives 1 point')
                            explosion.owner.update_score(s.reward_kill)
                            explosion.owner.step_events.append(e.KILLED_OPPONENT)
                            explosion.owner.trophies.append(smoothscale(a.avatar, (15, 15)))
            # Show smoke for a little longer
            if explosion.timer <= 0:
                explosion.active = False
            # Progress countdown
            explosion.timer -= 1
        for a in agents_hit:
            a.dead = True
            self.active_agents.remove(a)
            a.step_events.append(e.GOT_KILLED)
            for aa in self.active_agents:
                if aa is not a:
                    aa.step_events.append(e.OPPONENT_ELIMINATED)
            self.put_down_agent(a)
        self.explosions = [e for e in self.explosions if e.active]

        if self.agents[0].dead==False:# add visit
            trainer_agent_self = self.agents[0].get_state()
            x = trainer_agent_self[0]
            y = trainer_agent_self[1]
            self.trainer_visit = self.trainer_visit*0.9
            self.trainer_visit[x, y] += 1

        done = self.time_to_stop()
        observation = self.get_observation()
        reward = self.get_reward()
        self.state = self.get_info()
        info = self.state
        self.ep_reward += reward

        for a in self.agents:
            if a.record:
                    a.update_record(self.get_state_for_agent(a),done)
        if done:
            info['episode'] = {
                'r':self.ep_reward,
                'l':self.current_step,
                'visit':np.sum(self.trainer_visit>0),
                'coins':self.agents[0].score
            }
            self.end_round()
        return observation,reward, done, info

    def get_observation(self):
        return obser_fn(self.get_state_for_agent(self.agents[0]),self.agents[0].dead)
    def get_info(self):
        return self.get_state_for_agent(self.agents[0])
    def get_reward(self):
        return reward_fn(self.agents[0].step_events,self.get_info())
    def setup_logging(self):
        self.logger = logging.getLogger('BombeRLeWorld')
        self.logger.setLevel(s.log_game)
        handler = logging.FileHandler('logs/game.log', mode='w')
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info('Initializing game world')

    def setup_gui(self):
        # Initialize screen
        self.screen = pygame.display.set_mode((s.width, s.height))
        pygame.display.set_caption('BombeRLe')
        icon = pygame.image.load(f'assets/bomb_yellow.png')
        pygame.display.set_icon(icon)

        # Background and tiles
        self.background = pygame.Surface((s.width, s.height))
        self.background = self.background.convert()
        self.background.fill((0, 0, 0))
        self.t_wall = pygame.image.load('assets/brick.png')
        self.t_crate = pygame.image.load('assets/crate.png')

        # Font for scores and such
        font_name = 'assets/emulogic.ttf'
        self.fonts = {
            'huge': pygame.font.Font(font_name, 20),
            'big': pygame.font.Font(font_name, 16),
            'medium': pygame.font.Font(font_name, 10),
            'small': pygame.font.Font(font_name, 8),
        }

    def get_state_for_agent(self, agent, exit=False):
        state = {}
        state['step'] = self.current_step
        state['arena'] = np.array(self.arena)
        state['self'] = agent.get_state()
        state['train'] = agent.train_flag
        state['others'] = [other.get_state() for other in self.active_agents if other is not agent]
        state['bombs'] = [bomb.get_state() for bomb in self.bombs]
        state['coins'] = [coin.get_state() for coin in self.coins if coin.collectable]
        explosion_map = np.zeros(self.arena.shape)
        for e in self.explosions:
            for (x, y) in e.blast_coords:
                explosion_map[x, y] = max(explosion_map[x, y], e.timer)
        state['explosions'] = explosion_map
        state['exit'] = exit

        # the trace of trainer, only use for reward
        if type(agent)==BaseAgent:
            state['visit'] = self.trainer_visit
        return state

    def tile_is_free(self, x, y):
        is_free = (self.arena[x, y] == 0)
        if is_free:
            for obstacle in self.bombs + self.active_agents:
                is_free = is_free and (obstacle.x != x or obstacle.y != y)
        return is_free

    def perform_agent_action(self, agent, action):
        # Perform the specified action if possible, wait otherwise
        if action == 'UP' and self.tile_is_free(agent.x, agent.y - 1):
            agent.y -= 1
            agent.step_events.append(e.MOVED_UP)
        elif action == 'DOWN' and self.tile_is_free(agent.x, agent.y + 1):
            agent.y += 1
            agent.step_events.append(e.MOVED_DOWN)
        elif action == 'LEFT' and self.tile_is_free(agent.x - 1, agent.y):
            agent.x -= 1
            agent.step_events.append(e.MOVED_LEFT)
        elif action == 'RIGHT' and self.tile_is_free(agent.x + 1, agent.y):
            agent.x += 1
            agent.step_events.append(e.MOVED_RIGHT)
        elif action == 'BOMB' and agent.bombs_left > 0:
            self.logger.info(f'Agent <{agent.name}> drops bomb at {(agent.x, agent.y)}')
            self.bombs.append(agent.make_bomb())
            agent.bombs_left -= 1
            agent.step_events.append(e.BOMB_DROPPED)
        elif action == 'WAIT':
            agent.step_events.append(e.WAITED)
        else:
            agent.step_events.append(e.INVALID_ACTION)

    def poll_and_run_agents(self, input_action):
        # Perform decided agent actions
        perm = np.random.permutation(len(self.active_agents))
        actions = []
        for i in perm:
            a = self.active_agents[i]
            game_state = self.get_state_for_agent(a)
            if a.train_flag == True:
                action = input_action
            else:
                action = a.act(game_state)
            actions.append(action)
            self.logger.info(f'Agent <{a.name}> chose action {action}.')
        for i in range(len(perm)):
            action = actions[i]
            a = self.active_agents[perm[i]]
            self.perform_agent_action(a, action)

    def put_down_agent(self, agent):
        # Send exit message to end round for this agent
        self.logger.debug(f'Send exit message to end round for {agent.name}')
        agent.game_state = self.get_state_for_agent(agent, exit=True)

    def time_to_stop(self):
        # Check round stopping criteria
        if len(self.active_agents) == 0:
            self.logger.info(f'No agent left alive, wrap up round')
            return True
        if (len(self.active_agents) == 1
                and (self.arena == 1).sum() == 0
                and all([not c.collectable for c in self.coins])
                and len(self.bombs) + len(self.explosions) == 0):
            self.logger.info(f'One agent left alive with nothing to do, wrap up round')
            return True

        if s.stop_if_not_training:
            has_training = False
            for a in self.active_agents:
                if a.train_flag == True:
                    has_training = True
            if has_training == False:
                return True

        if self.current_step >= s.max_steps:
            self.logger.info('Maximum number of steps reached, wrap up round')
            return True

        return False

    def end_round(self):
        if self.running:
            # Wait in case there is still a game step running
            self.logger.info(f'WRAPPING UP ROUND #{self.round}')
            # Clean up survivors
            for a in self.active_agents:
                a.step_events.append(e.SURVIVED_ROUND)
                self.put_down_agent(a)
            # Send final event queue to agents that expect them
            for a in self.agents:
                if a.train_flag:
                    self.logger.debug(f'Sending final event queue {a.step_events} to agent <{a.name}>')
                    a.step_events = []
            # Mark round as ended
            self.running = False
        else:
            warnings.warn('End-of-round requested while no round was running')

    def close(self):
        if self.running:
            self.end_round()
            warnings.warn('SHUT DOWN')

    def render_text(self, text, x, y, color, halign='left', valign='top',
                    size='medium', aa=False):
        if not s.gui: return
        text_surface = self.fonts[size].render(text, aa, color)
        text_rect = text_surface.get_rect()
        if halign == 'left':   text_rect.left = x
        if halign == 'center': text_rect.centerx = x
        if halign == 'right':  text_rect.right = x
        if valign == 'top':    text_rect.top = y
        if valign == 'center': text_rect.centery = y
        if valign == 'bottom': text_rect.bottom = y
        self.screen.blit(text_surface, text_rect)

    def render(self):
        if not s.gui: return
        self.screen.blit(self.background, (0, 0))

        # World
        for x in range(self.arena.shape[1]):
            for y in range(self.arena.shape[0]):
                if self.arena[x, y] == -1:
                    self.screen.blit(self.t_wall,
                                     (s.grid_offset[0] + s.grid_size * x, s.grid_offset[1] + s.grid_size * y))
                if self.arena[x, y] == 1:
                    self.screen.blit(self.t_crate,
                                     (s.grid_offset[0] + s.grid_size * x, s.grid_offset[1] + s.grid_size * y))
        self.render_text(f'Step {self.current_step:d}', s.grid_offset[0], s.height - s.grid_offset[1] / 2, (64, 64, 64),
                         valign='center', halign='left', size='medium')

        # Items
        for bomb in self.bombs:
            bomb.render(self.screen, s.grid_offset[0] + s.grid_size * bomb.x, s.grid_offset[1] + s.grid_size * bomb.y)
        for coin in self.coins:
            if coin.collectable:
                coin.render(self.screen, s.grid_offset[0] + s.grid_size * coin.x,
                            s.grid_offset[1] + s.grid_size * coin.y)

        # Agents
        for agent in self.active_agents:
            agent.render(self.screen, s.grid_offset[0] + s.grid_size * agent.x,
                         s.grid_offset[1] + s.grid_size * agent.y)

        # Explosions
        for explosion in self.explosions:
            explosion.render(self.screen)

        # Scores
        # agents = sorted(self.agents, key=lambda a: (a.score, -a.mean_time), reverse=True)
        agents = self.agents
        leading = max(self.agents, key=lambda a: a.score)
        y_base = s.grid_offset[1] + 15
        for i, a in enumerate(agents):
            bounce = 0 if (a is not leading or self.running) else np.abs(10 * np.sin(5 * time()))
            a.render(self.screen, 600, y_base + 50 * i - 15 - bounce)
            self.render_text(a.name, 650, y_base + 50 * i,
                             (64, 64, 64) if a.dead else (255, 255, 255),
                             valign='center', size='small')
            for j, trophy in enumerate(a.trophies):
                self.screen.blit(trophy, (660 + 10 * j, y_base + 50 * i + 12))
            self.render_text(f'{a.score:d}', 830, y_base + 50 * i, (255, 255, 255),
                             valign='center', halign='right', size='big')
            self.render_text(f'{a.total_score:d}', 890, y_base + 50 * i, (64, 64, 64),
                             valign='center', halign='right', size='big')

        # End of round info
        if not self.running:
            x_center = (s.width - s.grid_offset[0] - s.cols * s.grid_size) / 2 + s.grid_offset[0] + s.cols * s.grid_size
            color = np.int_((255 * (np.sin(3 * time()) / 3 + .66),
                             255 * (np.sin(4 * time() + np.pi / 3) / 3 + .66),
                             255 * (np.sin(5 * time() - np.pi / 3) / 3 + .66)))
            self.render_text(leading.name, x_center, 320, color,
                             valign='top', halign='center', size='huge')
            self.render_text('has won the round!', x_center, 350, color,
                             valign='top', halign='center', size='big')
