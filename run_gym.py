import gym
import numpy as np

from stable_baselines.common.policies import CnnPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds
from env import BombeRLeWorld
import pickle

last_ep_rewmean = -np.inf
last_ep_lenmean = 1000
n_steps = 0
from settings import s

phase = s.phase


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, last_ep_rewmean, last_ep_lenmean
    # Print stats every 100 calls
    if (n_steps + 1) % 100 == 0:
        ep_info_buf = _locals['ep_info_buf']
        ep_rewmean = np.mean([ep_info['coins'] for ep_info in ep_info_buf])
        ep_lenmean = np.mean([ep_info['l'] for ep_info in ep_info_buf])
        if ep_rewmean > (last_ep_rewmean + 0.01):
            print('ep_rewmean increase from {:.4f} to {:.4f},save model'.format(last_ep_rewmean, ep_rewmean))
            last_ep_rewmean = ep_rewmean
            last_ep_lenmean = ep_lenmean
            _locals['self'].save(log_dir + 'ppo2_{}_best_r{:.4f}_l{:.4f}.pkl'.format(phase, ep_rewmean, ep_lenmean))
        # for coins and crates, the less the better
        elif phase in ['coins', 'crates75'] and np.abs(ep_rewmean - last_ep_rewmean) < 0.005 and (
                ep_lenmean < last_ep_lenmean - 5):
            print('ep_lenmean decrease from {:.4f} to {:.4f},save model'.format(last_ep_lenmean, ep_lenmean))
            last_ep_rewmean = ep_rewmean
            last_ep_lenmean = ep_lenmean
            _locals['self'].save(log_dir + 'ppo2_{}_best_r{:.4f}_l{:.4f}.pkl'.format(phase, ep_rewmean, ep_lenmean))
        elif np.abs(ep_rewmean - last_ep_rewmean) < 0.005 and (ep_lenmean > last_ep_lenmean + 5):
            print('ep_lenmean increase from {:.4f} to {:.4f},save model'.format(last_ep_lenmean, ep_lenmean))
            last_ep_rewmean = ep_rewmean
            last_ep_lenmean = ep_lenmean
            _locals['self'].save(log_dir + 'ppo2_{}_best_r{:.4f}_l{:.4f}.pkl'.format(phase, ep_rewmean, ep_lenmean))
    n_steps += 1
    return True


def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = BombeRLeWorld()
        return env

    set_global_seeds(seed + rank)
    return _init


if __name__ == '__main__':
    num_cpu = 4  # Number of processes to use
    log_dir = "./tmp/" + phase + '/'

    # Create the vectorized environment
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    model = PPO2(policy=CnnPolicy, env=env, gamma=0.99, n_steps=128,
                 nminibatches=4, vf_coef=0.5, ent_coef=0.01, noptepochs=4,
                 learning_rate=lambda f: f * 2.5e-4, cliprange=lambda f: f * 0.1,
                 verbose=1, tensorboard_log=log_dir)

    if phase == 'crates75' or phase == 'fight' or phase=='coins':
        restores = []
        with open('./tmp/imitation/policy_{}_disc0.99_weight10_best.weight'.format(phase), 'rb') as fp:
            weight_file = pickle.load(fp)

        for i in range(len(weight_file)):
            param = model.params[i]
            loaded_p = weight_file[i]
            restores.append(param.assign(loaded_p))
        model.sess.run(restores)
        print('save init model')
        model.save(log_dir + 'ppo2_{}_init.pkl'.format(phase))
    # elif phase=='fightraw':
    #     data, params = model._load_from_file(log_dir + 'ppo2_{}_best.pkl'.format(phase))
    #     restores = []
    #     for param, loaded_p in zip(model.params, params):
    #         restores.append(param.assign(loaded_p))
    #     model.sess.run(restores)
    # elif phase=='fight':
    #     data, params = model._load_from_file('./tmp/crates75/ppo2_crates75_best.pkl')
    #     restores = []
    #     for param, loaded_p in zip(model.params, params):
    #         restores.append(param.assign(loaded_p))
    #     model.sess.run(restores)

    # data, params = model._load_from_file('./tmp/coins/ppo2_coins_best.pkl')
    # restores = []
    #
    # # i=0
    # for param, loaded_p in zip(model.params, params):
    #     # if i<8:
    #         restores.append(param.assign(loaded_p))
    #         # i += 1
    # model.sess.run(restores)

    # model.load('./tmp/fight/ppo2_fight_best_r0.5617_l210.9600.pkl',env=env)

    # model = PPO2.load('./tmp/fight/ppo2_fight_best_r0.5617_l210.9600.pkl',env=env,tensorboard_log=log_dir,learning_rate=lambda f: f * 1e-3)
    model.learn(total_timesteps=int(1e7), callback=callback)
    model.save(log_dir + "ppo2_{}_final.pkl".format(phase))
