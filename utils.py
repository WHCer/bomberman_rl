from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from settings import e

def extract_scope(center_x, center_y, img, scope_size=1, padding_value=0):
    # img is (H,W)
    # scope_size 为马氏距离的中心周各添加
    pad_img = np.pad(img, ((scope_size, scope_size), (scope_size, scope_size)), 'constant',
                     constant_values=padding_value)
    scope = pad_img[center_y:center_y + scope_size * 2 + 1, center_x:center_x + scope_size * 2 + 1]
    return scope

def flatten_scope(scope, include_center=False):
    # flatten by distance order (left top to right down at each circle)
    # assert scope.shape[0] % 2 == 1 and scope.shape[1] == scope.shape[0]

    dis_indicater = np.zeros([1, 1], dtype=np.int)
    scope_size = scope.shape[0] // 2
    '''
    build distance indicator for index:
    scope_size=1, [[1,1,1],
                   [1,0,1],
                   [1,1,1]]
    scope_size=2, [[2,2,2,2,2],
                   [2,1,1,1,2],
                   [2,1,0,1,2],
                   [2,1,1,1,2],
                   [2,2,2,2,2]]
    '''
    # build distance indicator for index:
    # scope_size=1,
    #
    for dis in range(1, scope_size + 1):
        dis_indicater = np.pad(dis_indicater, ((1, 1), (1, 1)), 'constant', constant_values=dis)
    flat_feature = np.empty([0])
    if include_center:
        flat_feature = np.append(flat_feature, scope[dis_indicater == 0])

    for dis in range(1, scope_size + 1):
        flat_feature = np.append(flat_feature, scope[dis_indicater == dis])
    return flat_feature

def game_state_to_img(game_state,dead=False):
    """
    convert game_state to a 1 channel image, Different value represent different meaning
    H=17,W=17 in original
    :param game_state:
    :return:
        :param img np.float32 array with shape (H,W,1) range [0,1]
         # 2 for crates , 0 fr stone, 1 for free tiles
         # 3 for coins
         # if not die:
             # 4 for self with booms
             # 5 for self without booms
         # 6 for all others
         # 7 for boom
         # 8 for explosions
         # 10 if self sit on a boom!!
         then divided by 8.0 to normalized to [0,1]
    """
    img = game_state['arena'].copy().astype(np.float32).T  # 1 for crates , -1 fr stone, 0 for free tiles
    img += 1.0  # 2 for crates , 0 fr stone, 1 for free tiles

    # 3 for coins
    for coin in game_state['coins']:
        img[coin[1], coin[0]] = 3

    if not dead:
        self = game_state['self']  # (x,y,name,has_boom{0,1},score)
        # 4 for self with booms, 5 for self without booms
        img[self[1], self[0]] = 4 + (1-self[3])

    # 6 for all others
    for other in game_state['others']:
        img[other[1], other[0]] =6

    # 7for boom
    for bomb in game_state['bombs']:
        # print("bomb count{}".format(12 - bomb[-1]))
        img[bomb[1], bomb[0]] = 7
        self = game_state['self']
        if self[0]==bomb[0] and self[1]==bomb[1]:
            img[bomb[1], bomb[0]] = 10
    # 8 for explosion
    explosions = game_state['explosions'].T.copy()
    explosions_index = explosions > 0
    # print("explosions: {}".format(np.unique(explosions[explosions_index] + 12.0)))
    img[explosions_index] = 8.0

    # normalize base_arena
    # print("index:{}".format(np.unique(base_arena)))

    img = img / 10.0
    img = np.array(img, dtype=np.float32)
    # img = np.expand_dims(base_arena,-1)
    # print("max:{}".format(np.max(base_arena)))
    return np.expand_dims(img,axis=-1)


def game_state_to_img_channel(game_state,dead):
    """
    convert game_state to a 1 channel image, Different value represent different meaning
    H=17,W=17 in original
    :param game_state:
    :return:
        :param img np.float32 array with shape (H,W,C) range [0,1] C=7
         # channel 1: 0 others 1 for stone
         # channel 2: 0 others, 1 for free tiles
         # channel 3: 0 others, 1 for crates
         # channel 4: 0 for others 0.5 for self without booms, 1 for self with boom
         # channel 5: 0 for others 0.5 for all other components without booms, 1 for all others witm boom
         # channel 6: 0 for others 1 for boom
         # channel 7: 0 for others, 1 for explosions
    """

    img = game_state['arena'].copy().T  # 1 for crates , -1 fr stone, 0 for free tiles

    # stone
    stone_map = np.zeros_like(img,dtype=np.float32)
    stone_map[img==-1] = 1
    # free tiles
    free_map = np.zeros_like(img,dtype=np.float32)
    free_map[img==0] = 1
    # crates
    crates_map = np.zeros_like(img,dtype=np.float32)
    crates_map[img==1] = 1
    # self
    self_map = np.zeros_like(img,dtype=np.float32)
    if not dead:
        self = game_state['self']  # (x,y,name,has_boom{0,1},score)
        if self[3]==0:
            self_map[self[1], self[0]] = 0.5
            free_map[self[1], self[0]] = 0
        else:
            self_map[self[1], self[0]] = 1
            free_map[self[1], self[0]] = 0
    # others
    others_map = np.zeros_like(img,dtype=np.float32)
    for other in game_state['others']:
        others_map[other[1], other[0]] = 0.5 + 0.5*(other[3])
        free_map[other[1], other[0]] = 0
    # boom
    boom_map = np.zeros_like(img,dtype=np.float32)
    for bomb in game_state['bombs']:
        # print("bomb count{}".format(12 - bomb[-1]))
        boom_map[bomb[1], bomb[0]] = 1
        free_map[bomb[1], bomb[0]] = 0
    # exp
    explosions_map = np.zeros_like(img,dtype=np.float32)
    explosions = game_state['explosions'].T.copy()
    explosions_index = explosions > 0
    explosions_map[explosions_index] = 1
    free_map[explosions_index] = 0


    img_channel = np.stack((stone_map,free_map,crates_map,self_map,others_map,boom_map,explosions_map),axis=-1)
    return img_channel


def game_state_to_img_complex(game_state):
    """
    convert game_state to a 1 channel image, Different value represent different meaning
    H=17,W=17 in original
    :param game_state:
    :return:
        :param img np.float32 array with shape (H,W) range [0,1]
         # 2 for crates , 0 fr stone, 1 for free tiles
         # 3 for coins
         # 4 for self without BOMB, 5 for self with BOMB
         # 6 for all others without BOMB, 7 for all others with BOMB
         # 8 for boom with count downs 0,  9,10,11,12 for 1,2,3,4   -> count number for 4 (0,1,2,3,4)
         # 13 for explosions with present time 1, 14 for 2->  present time for 2
         then divided by 14.0 to normalized to [0,1]
    """
    img = game_state['arena'].copy().astype(np.float32).T  # 1 for crates , -1 fr stone, 0 for free tiles
    img += 1.0  # 2 for crates , 0 fr stone, 1 for free tiles

    # 3 for coins
    for coin in game_state['coins']:
        img[coin[1], coin[0]] = 3

    self = game_state['self']  # (x,y,name,has_boom{0,1},score)
    # 4 for self without BOMB, 5 for self with BOMB
    img[self[1], self[0]] = 4 + self[3]

    # 6 for all others without BOMB, 7 for all others with BOMB
    for other in game_state['others']:
        img[other[1], other[0]] = 6 + other[3]

    # (12,11,10,9,8) for boom with count down (0,1,2,3,4) -> count number for 4 (0,1,2,3,4)
    for bomb in game_state['bombs']:
        # print("bomb count{}".format(12 - bomb[-1]))
        img[bomb[1], bomb[0]] = 12 - bomb[-1]
    # (14,13) for explosions with present time (1,0)  -> present time for 2(0,1)
    explosions = game_state['explosions'].T
    explosions_index = explosions > 0
    # print("explosions: {}".format(np.unique(explosions[explosions_index] + 12.0)))
    img[explosions_index] = explosions[explosions_index] + 12.0

    # normalize base_arena
    # print("index:{}".format(np.unique(base_arena)))
    img = img / 14.0
    img = np.array(img, dtype=np.float32)
    # img = np.expand_dims(base_arena,-1)
    # print("max:{}".format(np.max(base_arena)))
    return img


def game_state_to_flat_feature(game_state, scope_size=6):
    img = game_state_to_img(game_state)
    self = game_state['self']  # (x,y,name,has_boom{0,1},score)

    flat_self_feature = np.array([self[0], self[1],self[3]])
    scope = extract_scope(center_x=self[0], center_y=self[1], img=img, scope_size=scope_size, padding_value=0)
    flat_scope_feature = flatten_scope(scope, include_center=True)

    # coins location feature
    cols = img.shape[1]
    rows = img.shape[0]
    n_split = 3
    flat_coins_feature = np.zeros((n_split * n_split, 2))
    x_len = (cols - 2) // n_split
    y_len = (rows - 2) // n_split
    assert (cols - 2) % n_split == 0 and (rows - 2) % n_split == 0
    for i in range(n_split):
        for j in range(n_split):
            xmin = 1 + x_len * j
            xmax = 1 + x_len * (j + 1)
            ymin = 1 + y_len * i
            ymax = 1 + y_len * (i + 1)
            coins_x = 0
            coins_y = 0
            for coin in game_state['coins']:
                x, y = coin[0], coin[1]
                if xmin <= x and x < xmax and ymin <= y and y < ymax:
                    coins_x = x
                    coins_y = y
                    break
            flat_coins_feature[i * n_split + j] = [coins_x, coins_y]
    flat_coins_feature = flat_coins_feature.flatten()

    flat_feature = np.concatenate((flat_self_feature, flat_scope_feature, flat_coins_feature))
    return flat_feature.astype(np.float32)

## reward use for training collect coins
def get_reward_coins(events,game_state):
    reward = 0
    if e.COIN_COLLECTED in events:  # Collected a coin.
        reward += 1
    if e.GOT_KILLED in events:
        reward += -5
    return reward / 5

count_with_boom = 0
def get_reward_crates(events,game_state):
    global count_with_boom
    reward = 0
    # if there is crates within 2 scope size, but you hold a boom, reward decrease
    # if there is component within 2 scope size but you hold a boom,reward decrease
    self = game_state['self']
    if self[3]==1: # has boom
        base = game_state['arena'].copy().T # 1 for crates -1 for stone 0 for free tile
        base[base<0] = 0 #1 for crates 0 for others
        self = game_state['self']
        scope = extract_scope(self[0],self[1],base,scope_size=2,padding_value=0)
        if np.sum(scope)>1:
            # has crates nearby, you should use your boom
            count_with_boom +=1
            if count_with_boom>20:
                reward -= 0.5
    else:
        count_with_boom=0
    # walk to new block will receive reward
    visit =game_state['visit'][self[0],self[1]]
    if visit==1:
        reward+=0.5
    elif visit>1:
        reward -= 0.02*(visit-1)

    if e.COIN_COLLECTED in events:  # Collected a coin.
        reward += 2
    if e.CRATE_DESTROYED in events:
        reward += 0.5
    if e.COIN_FOUND in events:
        reward += 0.5
    if e.GOT_KILLED in events:
        reward += -5
    return reward / 5
def get_reward_fight(events,game_state):
    global count_with_boom
    reward = 0
    # if there is crates within 2 scope size, but you hold a boom, reward decrease
    self = game_state['self']
    if self[3] == 1:  # has boom
        base = game_state['arena'].copy().T  # 1 for crates -1 for stone 0 for free tile
        base[base < 0] = 0  # 1 for crates 0 for others
        for other in game_state['others']:
            base[other[1], other[0]] =2
        self = game_state['self']
        scope = extract_scope(self[0], self[1], base, scope_size=2, padding_value=0)
        if np.sum(scope) > 1:
            # has crates nearby, you should use your boom
            count_with_boom += 1
            if 2 in scope:
                count_with_boom +=1
            if count_with_boom > 20:
                reward -= 0.1
    else:
        count_with_boom = 0
    # walk to new block will receive reward
    if e.COIN_COLLECTED in events:  # Collected a coin.
        reward += 5
    if e.KILLED_OPPONENT in events:  # Blew up an opponent.
        reward += 10
    if e.GOT_KILLED in events:
        reward += -5
    return reward / 10


# from random import shuffle
# from collections import deque
#
#
# def look_for_targets(free_space, start, targets, logger=None):
#     """Find direction of closest target that can be reached via free tiles.
#
#     Performs a breadth-first search of the reachable free tiles until a target is encountered.
#     If no target can be reached, the path that takes the agent closest to any target is chosen.
#
#     Args:
#         free_space: Boolean numpy array. True for free tiles and False for obstacles.
#         start: the coordinate from which to begin the search.
#         targets: list or array holding the coordinates of all target tiles.
#         logger: optional logger object for debugging.
#     Returns:
#         coordinate of first step towards closest target or towards tile closest to any target.
#     """
#     if len(targets) == 0: return None
#
#     frontier = [start]
#     parent_dict = {start: start}
#     dist_so_far = {start: 0}
#     best = start
#     best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()
#
#     while len(frontier) > 0:
#         current = frontier.pop(0)
#         # Find distance from current position to all targets, track closest
#         d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
#         if d + dist_so_far[current] <= best_dist:
#             best = current
#             best_dist = d + dist_so_far[current]
#         if d == 0:
#             # Found path to a target's exact position, mission accomplished!
#             best = current
#             break
#         # Add unexplored free neighboring tiles to the queue in a random order
#         x, y = current
#         neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
#         shuffle(neighbors)
#         for neighbor in neighbors:
#             if neighbor not in parent_dict:
#                 frontier.append(neighbor)
#                 parent_dict[neighbor] = current
#                 dist_so_far[neighbor] = dist_so_far[current] + 1
#     if logger: logger.debug(f'Suitable target found at {best}')
#     # Determine the first step towards the best found target tile
#     current = best
#     while True:
#         if parent_dict[current] == start: return current
#         current = parent_dict[current]
#
#
# def simple_agent_setup(self):
#     """Called once before a set of games to initialize data structures etc.
#
#     The 'self' object passed to this method will be the same in all other
#     callback methods. You can assign new properties (like bomb_history below)
#     here or later on and they will be persistent even across multiple games.
#     You can also use the self.logger object at any time to write to the log
#     file for debugging (see https://docs.python.org/3.7/library/logging.html).
#     """
#     self.logger.debug('Successfully entered setup code')
#     np.random.seed()
#     # Fixed length FIFO queues to avoid repeating the same actions
#     self.bomb_history = deque([], 5)
#     self.coordinate_history = deque([], 20)
#     # While this timer is positive, agent will not hunt/attack opponents
#     self.ignore_others_timer = 0
#
#
# def simple_agent_act(self):
#     """Called each game step to determine the agent's next action.
#
#     You can find out about the state of the game environment via self.game_state,
#     which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
#     what it contains.
#
#     Set the action you wish to perform by assigning the relevant string to
#     self.next_action. You can assign to this variable multiple times during
#     your computations. If this method takes longer than the time limit specified
#     in settings.py, execution is interrupted by the game and the current value
#     of self.next_action will be used. The default value is 'WAIT'.
#     """
#     # Gather information about the game state
#     arena = self.game_state['arena']
#     x, y, _, bombs_left, score = self.game_state['self']
#     bombs = self.game_state['bombs']
#     bomb_xys = [(x, y) for (x, y, t) in bombs]
#     others = [(x, y) for (x, y, n, b, s) in self.game_state['others']]
#     coins = self.game_state['coins']
#     bomb_map = np.ones(arena.shape) * 5
#     for xb, yb, t in bombs:
#         for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
#             if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
#                 bomb_map[i, j] = min(bomb_map[i, j], t)
#
#     # If agent has been in the same location three times recently, it's a loop
#     if self.coordinate_history.count((x, y)) > 2:
#         self.ignore_others_timer = 5
#     else:
#         self.ignore_others_timer -= 1
#     self.coordinate_history.append((x, y))
#
#     # Check which moves make sense at all
#     directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
#     valid_tiles, valid_actions = [], []
#     for d in directions:
#         if ((arena[d] == 0) and
#                 (self.game_state['explosions'][d] <= 1) and
#                 (bomb_map[d] > 0) and
#                 (not d in others) and
#                 (not d in bomb_xys)):
#             valid_tiles.append(d)
#     if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
#     if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
#     if (x, y - 1) in valid_tiles: valid_actions.append('UP')
#     if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
#     if (x, y) in valid_tiles: valid_actions.append('WAIT')
#     # Disallow the BOMB action if agent dropped a bomb in the same spot recently
#     if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
#
#     # Collect basic action proposals in a queue
#     # Later on, the last added action that is also valid will be chosen
#     action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
#     shuffle(action_ideas)
#
#     # Compile a list of 'targets' the agent should head towards
#     dead_ends = [(x, y) for x in range(1, s.cols - 1) for y in range(1, s.rows - 1) if (arena[x, y] == 0)
#                  and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
#     crates = [(x, y) for x in range(1, s.cols - 1) for y in range(1, s.rows - 1) if (arena[x, y] == 1)]
#     targets = coins + dead_ends + crates
#     # Add other agents as targets if in hunting mode or no crates/coins left
#     if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
#         targets.extend(others)
#
#     # Exclude targets that are currently occupied by a bomb
#     targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]
#
#     # Take a step towards the most immediately interesting target
#     free_space = arena == 0
#     if self.ignore_others_timer > 0:
#         for o in others:
#             free_space[o] = False
#     d = look_for_targets(free_space, (x, y), targets)
#     if d == (x, y - 1): action_ideas.append('UP')
#     if d == (x, y + 1): action_ideas.append('DOWN')
#     if d == (x - 1, y): action_ideas.append('LEFT')
#     if d == (x + 1, y): action_ideas.append('RIGHT')
#     if d is None:
#         self.logger.debug('All targets gone, nothing to do anymore')
#         action_ideas.append('WAIT')
#
#     # Add proposal to drop a bomb if at dead end
#     if (x, y) in dead_ends:
#         action_ideas.append('BOMB')
#     # Add proposal to drop a bomb if touching an opponent
#     if len(others) > 0:
#         if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
#             action_ideas.append('BOMB')
#     # Add proposal to drop a bomb if arrived at target and touching crate
#     if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
#         action_ideas.append('BOMB')
#
#     # Add proposal to run away from any nearby bomb about to blow
#     for xb, yb, t in bombs:
#         if (xb == x) and (abs(yb - y) < 4):
#             # Run away
#             if (yb > y): action_ideas.append('UP')
#             if (yb < y): action_ideas.append('DOWN')
#             # If possible, turn a corner
#             action_ideas.append('LEFT')
#             action_ideas.append('RIGHT')
#         if (yb == y) and (abs(xb - x) < 4):
#             # Run away
#             if (xb > x): action_ideas.append('LEFT')
#             if (xb < x): action_ideas.append('RIGHT')
#             # If possible, turn a corner
#             action_ideas.append('UP')
#             action_ideas.append('DOWN')
#     # Try random direction if directly on top of a bomb
#     for xb, yb, t in bombs:
#         if xb == x and yb == y:
#             action_ideas.extend(action_ideas[:4])
#
#     # Pick last action added to the proposals list that is also valid
#     while len(action_ideas) > 0:
#         a = action_ideas.pop()
#         if a in valid_actions:
#             self.next_action = a
#             break
#
#     # Keep track of chosen action for cycle detection
#     if self.next_action == 'BOMB':
#         self.bomb_history.append((x, y))
