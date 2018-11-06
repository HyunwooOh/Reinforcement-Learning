# Reinforcement Learning
- Tensorflow implementation of Reinforcement Learning methods for Atari 2600.
- 강화학습의 유명한 알고리즘들을 구현한 저장소입니다.
- Atari 2600 를 대상으로 실험할 수 있습니다.

## Requirements
- Python v3.6
- tensorflow v1.4
- OpenAI Gym v0.9

## How to use
- `python3 main.py --game [game name] --model [model name]`
    - [game name] : BreakoutDeterministic, Seaquest, etc. default : BreakoutDeterministic
    - [model name] : dqn, a3c etc. default : dqn

## Contents
- Value Based Reinforcement Learning
    - [DQN](dqn.py)
    - [Double DQN](./utils/valuebase_utils.py)
    - [Dueling DQN](dqn.py)
    - [DRQN](drqn.py)
    - [DRQN with training initial state](drqn.py)
- Policy Based Reinforcement Learning
    - [A3C](a3c.py)
- To be added
    - Value Based RL
        - Asynchronous n-step Q-learning
    - Policy Based RL
        - PG, DPG, DDPG
        - NPG, TRPO, GAE, PPO
    - with Visual Attention Mechanism
        - DARQN (DRQN with Visaul Attention)

## References
- Studies
    - [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236/)
    - [Deep Reinforcement Learning with Double Q-Learning](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12389/11847)
    - [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
    - [Deep recurrent q-learning for partially observable mdps](http://www.aaai.org/ocs/index.php/FSS/FSS15/paper/download/11673/11503)
    - [Deep attention recurrent Q-network](https://arxiv.org/abs/1512.01693)
    - [Asynchronous Methods for Deep Reinforcement Learning](http://proceedings.mlr.press/v48/mniha16.pdf)
- Github repositories
    - https://gist.github.com/awjuliani/35d2ab3409fc818011b6519f0f1629df