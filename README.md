# Reinforcement Learning
- Tensorflow implementations of Reinforcement Learning methods for Atari 2600.
- 강화학습의 유명한 알고리즘들을 구현했습니다.
- Atari 2600 를 대상으로 실험할 수 있습니다.

## Requirements
- Python v3.6
- tensorflow v1.4
- OpenAI Gym v0.9

## How to use
- `python3 main.py --game [game name] --model [model name]`
    - [game name] : BreakoutDeterministic, Seaquest, etc. default : BreakoutDeterministic
    - [model name] : dqn, a3c etc. default : dqn
- 자세한 설정은 [main.py](main.py), [Config.py](./utils/Config.py) 에서 조정할 수 있습니다.
- [dqn.py](dqn.py), [drqn.py](drqn.py)는 네트워크 클래스와 아타리에서의 학습을 할 수 있는 메소드로 이루어져 있습니다.
- [a3c.py](a3c.py)는 네트워크 클래스와 아타리에서 학습을 진행하는 Worker 클래스, global 네트워크와 worker를 설정하고 학습을 시작하는 메소드로 이루어져 있습니다.

## Contents
- Value Based Reinforcement Learning
    - [DQN](dqn.py)
    - [Double DQN](./utils/valuebase_utils.py)
    - [Dueling DQN](dqn.py)
    - [DRQN](drqn.py)
- Policy Based Reinforcement Learning
    - [A3C](a3c.py)
- To be added
    - DARQN
    - Asynchronous n-step Q-learning
    - PG, DPG, DDPG
    - NPG, TRPO, GAE, PPO

## Note
- [NOTE.md](NOTE.md) 에 비고사항을 기록했습니다.

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
    - https://github.com/openai/universe-starter-agent/blob/293904f01b4180ecf92dd9536284548108074a44/a3c.py
    - https://github.com/rlcode/reinforcement-learning-kr
    