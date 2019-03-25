import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from stable_baselines import PPO2
from settings import s
ACTION_SET = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
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
def setup(self):
    self.model = PPO2.load('fight.pkl')
    fack_action = self.model.predict(np.zeros((17,17,1))) # fack action to activate agent
def act(self):
    obs = game_state_to_img(self.game_state)
    action, _states = self.model.predict(obs)
    self.next_action = ACTION_SET[action]

def reward_update(self):
    pass

def end_of_episode(self):
    pass
