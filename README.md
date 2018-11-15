# Reinforcement Learning
- Tensorflow implementations of Reinforcement Learning methods for Atari 2600.
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
- 자세한 설정은 [main.py](main.py), [Config.py](./utils/Config.py) 에서 조정할 수 있습니다.

## Contents
- Value Based Reinforcement Learning
    - [DQN](dqn.py)
    - [Double DQN](./utils/valuebase_utils.py)
    - [Dueling DQN](dqn.py)
    - [DRQN](drqn.py)
- Policy Based Reinforcement Learning
    - [A3C](a3c.py)
- To be added
    - Value Based RL
        - DARQN
        - Asynchronous n-step Q-learning
    - Policy Based RL
        - PG, DPG, DDPG
        - NPG, TRPO, GAE, PPO

## Note
### A3C's loss function
- I implemented Actor and Critic not as separated networks but as one network.
- To do this, i combined Actor's loss and Critic's loss together into one loss function.
- I found mainly two types of total loss function.
- [A3C논문](http://proceedings.mlr.press/v48/mniha16.pdf)에서 Actor와 Critic을 별개의 네트워크로 분리시켰습니다.
- 다른 분들의 코드를 보면 Actor와 Critic을 하나의 네트워크로 구성했습니다. 이를 위해 Actor의 손실함수와 Critic의 손실함수를 하나의 손실함수로 표현해야 합니다. 주로 두가지 형태를 찾을 수 있었습니다.
    - total_loss = 0.5 * critic_loss - actor_loss + 0.01 * entropy
    - total_loss = 0.5 * 0.5 * critic_loss - actor_loss + 0.01 * entropy
    - <img src= "/assets/a3c_loss_experiment.png" width="100%" height="100%">
    - BreakoutDeterministic-v4 에서 실험한 결과 0.5 * 0.5 * critic_loss 가 더 좋은 성능이 나왔습니다.
- [A3C를 스타크래프2에 사용한 논문](https://arxiv.org/pdf/1708.04782.pdf)의 4.1장에 그와 관련된 내용이 있습니다.

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
    - https://gist.github.com/jcwleo/fffc40f69b7f14d9e2a2b8765a79b579#file-dqn_breakout-py
    - https://github.com/rlcode/reinforcement-learning-kr
    