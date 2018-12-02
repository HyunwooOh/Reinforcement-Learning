# Note
## A3C's loss function
- I implemented Actor and Critic not as separated networks but as one network.
- To do this, i combined Actor's loss and Critic's loss together into one loss function.
- I found mainly two types of total loss function.
- [A3C논문](http://proceedings.mlr.press/v48/mniha16.pdf)에서 Actor와 Critic을 별개의 네트워크로 분리시켰습니다.
- Actor와 Critic을 하나의 네트워크로 구성했습니다. 이를 위해 분리되어 있던 Actor의 손실함수와 Critic의 손실함수를 하나의 손실함수로 표현해야 합니다. 주로 두가지 조합을 찾을 수 있었습니다.
    - `total_loss = 0.5 * critic_loss - actor_loss + 0.01 * entropy`
    - `total_loss = 0.5 * 0.5 * critic_loss - actor_loss + 0.01 * entropy`
    - <img src= "/assets/a3c_loss_experiment.png" width="100%" height="100%">
    - BreakoutDeterministic-v4 에서 실험한 결과 `0.5 * 0.5 * critic_loss` 의 성능이 더 좋았습니다.
- [A3C를 스타크래프2에 사용한 논문](https://arxiv.org/pdf/1708.04782.pdf)의 4.1장에 그와 관련된 내용이 있습니다.
