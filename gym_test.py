import os
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from stable_baselines import PPO2

from env import BombeRLeWorld

from settings import s
import contextlib

with contextlib.redirect_stdout(None):
    import pygame
from pygame.locals import *
from time import time, sleep

from tqdm import tqdm
phase = s.phase
if __name__ == '__main__':
    pygame.init()
    model = PPO2.load('./tmp/{}/ppo2_{}_init.pkl'.format(phase,phase))
    env = BombeRLeWorld()
    for round in tqdm(range(100)):
        start = time()
        step = 0
        obs = env.reset()
        if s.gui:
            sleep(1)
            # plt.imshow(obs)
            # plt.show()
            env.render()
            pygame.display.flip()
        while (True):
            step +=1
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            if s.gui:
                sleep(1 / 20)
                env.render()
                pygame.display.flip()
            if done:
                print('round{}----round reward:{},score{},step:{},done:{}'.format(round, info['episode']['r'],env.agents[0].score,info['episode']['l'],
                                                                          done))
                print('each step takes:{:.4f}'.format((time()-start)/step))
                break


    # model.learn(total_timesteps=int(1e7),callback=callback)
    # model.save(log_dir+"ppo2_{}_final.pkl".format(phase))