import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds
def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init
if __name__ == '__main__':
    env_id = "CartPole-v1"
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    #
    # env = gym.make('CartPole-v1')
    # env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

    model = PPO2(policy=MlpPolicy, env=env,gamma=0.99,n_steps=2048,nminibatches=32,
                 lam=0.95,noptepochs=10,ent_coef=0.0,learning_rate=3e-4,cliprange=0.2,
                 verbose=1)
    model.learn(total_timesteps=150000)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()