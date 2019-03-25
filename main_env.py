from time import time, sleep
import contextlib
from time import time

with contextlib.redirect_stdout(None):
    import pygame
from pygame.locals import *
import numpy as np
import multiprocessing as mp
import threading

import matplotlib.pyplot as plt
ACTION_TO_VALUE = {'UP':0, 'DOWN':1, 'LEFT':2, 'RIGHT':3, 'BOMB':4, 'WAIT':5}
from tqdm import  tqdm
from env import BombeRLeWorld,SimpleAgent
from settings import s
if __name__ == '__main__':
    pygame.init()
    env = BombeRLeWorld()
    agent = SimpleAgent('0','blue',True)
    for round in tqdm(range(10000)):
        agent.reset(round)
        observation = env.reset()

        if s.gui:
            sleep(1)
            # plt.imshow(obs)
            # plt.show()
            env.render()
            pygame.display.flip()
            # for i in range(observation.shape[-1]):
            #     plt.imshow(observation[:,:,i])
            #     plt.show()
        while (True):
            #
            action = agent.act(env.state)
            action = ACTION_TO_VALUE[action]
            observation, reward, done, info = env.step(action)
            if s.gui:
                sleep(0.1)
                env.render()
                pygame.display.flip()
                # for i in range(observation.shape[-1]):
                #     plt.imshow(observation[:, :, i])
                #     plt.show()
            if done:
                if s.gui:
                    env.render()
                    pygame.display.flip()
                print('round{}----reward:{},round score{},done:{}'.format(round,info['episode']['r'],env.agents[0].score,done))
                break
