# bomberman_rl
Solution for FML course final project from Hao and Yebei.

Our trained agent code are located in ./agent_code/my_agent 
Weights for each task are in release1.0.

# Performance 
We evaluate our agent for each task by taking average scores of 100 round. Note: slowest punishment is ignored.

For coins collection, it takes around 66 steps to collect all agents.

For crates destroy, it collects 6.5 coins per round, while simple agent collects 7.83 coins per round.

For play against 3 simple agents, it gains around 1.6-3.4 score per round, while simple agent gains 3.5-4.3 score per round.

# Others
We modify the default enviroment to fit the general Gym to run in parallel model. Refer to env.py. https://github.com/WHCer/bomberman_rl/blob/master/env.py

Our observation setting and rewarding shaping is in
https://github.com/WHCer/bomberman_rl/blob/master/utils.py

Our Imitation learning settingis in 
https://github.com/WHCer/bomberman_rl/blob/master/train_imitation.py

The modified network structure is in 
https://github.com/WHCer/bomberman_rl/blob/fe869198a5df03b03a8ae9e89fd166927ea30447/agent_code/my_agent/stable_baselines/common/policies.py#L15

# Requirements
As we use the PPO2 of https://stable-baselines.readthedocs.io/en/master/index.html, the requirements of this package is needed.
We use a modified PPO2, so simply install the stable-baselines directed might don't work.
The hyper-parameters are listed on https://github.com/araffin/rl-baselines-zoo/blob/b76641ea90e9e8eaf485ad1aee6dde272fa2470b/hyperparams/ppo2.yml#L1


