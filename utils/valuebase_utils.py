import numpy as np
import random
from utils.common_utils import input_image, input_image_drqn
from utils.Config import Config

##############################
########### POLICY ###########
##############################
def epsilon_greedy_policy(epsilon, ACTION_SIZE, Q):
    if epsilon > np.random.rand(1):
        action = np.random.randint(ACTION_SIZE)
    else:
        action = np.argmax(Q)
    return action
###############################
########### For DQN ###########
###############################
def train_dqn(sess, main, target, memory):
    minibatch = random.sample(memory, Config.BATCH_SIZE)
    histories = np.vstack([x[0][:, :, :, :Config.NUM_FRAME] for x in minibatch])
    actions = np.array([x[1] for x in minibatch])
    rewards = np.array([x[2] for x in minibatch])
    next_histories = np.vstack([x[0][:, :, :, 1:] for x in minibatch])
    deads = np.array([x[3] for x in minibatch])
    targetQ = rewards + 0.99 * np.max(sess.run(target.Qout, feed_dict={target.input:input_image(next_histories), target.batch_size:Config.BATCH_SIZE}), axis=1) * ~deads
    sess.run(main.optimize, feed_dict={main.input:input_image(histories), main.batch_size:Config.BATCH_SIZE, main.targetQ:targetQ, main.actions:actions})

def train_double_dqn(sess, main, target, memory):
    minibatch = random.sample(memory, Config.BATCH_SIZE)
    histories = np.vstack([x[0][:, :, :, :Config.NUM_FRAME] for x in minibatch])
    actions = np.array([x[1] for x in minibatch])
    rewards = np.array([x[2] for x in minibatch])
    next_histories = np.vstack([x[0][:, :, :, 1:] for x in minibatch])
    deads = np.array([x[3] for x in minibatch])
    Q1 = np.argmax(sess.run(main.Qout, feed_dict={main.input:input_image(next_histories), main.batch_size:Config.BATCH_SIZE}), 1)
    Q2 = sess.run(target.Qout, feed_dict={target.input:input_image(next_histories), target.batch_size:Config.BATCH_SIZE})
    doubleQ = Q2[range(Config.BATCH_SIZE), Q1]
    targetQ = rewards + 0.99 * doubleQ * ~deads
    sess.run(main.optimize, feed_dict={main.input:input_image(histories), main.batch_size:Config.BATCH_SIZE, main.targetQ:targetQ, main.actions:actions})

################################
########### For DRQN ###########
################################
def drqn_sampling(memory):
    sampled_episodes = random.sample(memory, Config.DRQN_BATCH_SIZE)
    sampled_traces = []
    for episode in sampled_episodes:
        point = np.random.randint(0, len(episode) + 1 - Config.UNROLLING_TIME_STEPS)
        sampled_traces.append(episode[point:point + Config.UNROLLING_TIME_STEPS])
    sampled_traces = np.array(sampled_traces)
    return np.reshape(sampled_traces, [Config.DRQN_BATCH_SIZE * Config.UNROLLING_TIME_STEPS, 5])

def train_drqn(sess, main, target, memory):
    minibatch = drqn_sampling(memory)
    states = np.vstack([x[0] for x in minibatch])
    actions = np.array([x[1] for x in minibatch])
    rewards = np.array([x[2] for x in minibatch])
    next_states = np.vstack([x[3] for x in minibatch])
    deads = np.array([x[4] for x in minibatch])
    state_train = (np.zeros([Config.DRQN_BATCH_SIZE, Config.DRQN_HSIZE]), np.zeros([Config.DRQN_BATCH_SIZE, Config.DRQN_HSIZE]))
    Q = sess.run(target.Qout, feed_dict = {target.input: input_image_drqn(next_states), target.batch_size: Config.DRQN_BATCH_SIZE, target.unrolling_time_steps: Config.UNROLLING_TIME_STEPS, target.state_in: state_train})
    targetQ = rewards + 0.99 * np.max(Q, axis=1) * ~deads
    sess.run(main.optimize, feed_dict={main.input: input_image_drqn(states), main.batch_size: Config.DRQN_BATCH_SIZE, main.unrolling_time_steps: Config.UNROLLING_TIME_STEPS, main.state_in: state_train, main.targetQ: targetQ, main.actions: actions})

def train_double_drqn(sess, main, target, memory):
    minibatch = drqn_sampling(memory)
    states = np.vstack([x[0] for x in minibatch])
    actions = np.array([x[1] for x in minibatch])
    rewards = np.array([x[2] for x in minibatch])
    next_states = np.vstack([x[3] for x in minibatch])
    deads = np.array([x[4] for x in minibatch])
    state_train = (np.zeros([Config.DRQN_BATCH_SIZE, Config.DRQN_HSIZE]), np.zeros([Config.DRQN_BATCH_SIZE, Config.DRQN_HSIZE]))
    Q1 = np.argmax(sess.run(main.Qout, feed_dict={main.input: input_image_drqn(next_states), main.batch_size: Config.DRQN_BATCH_SIZE, main.unrolling_time_steps: Config.UNROLLING_TIME_STEPS, main.state_in: state_train}), 1)
    Q2 = sess.run(target.Qout, feed_dict = {target.input: input_image_drqn(next_states), target.batch_size: Config.DRQN_BATCH_SIZE, target.unrolling_time_steps: Config.UNROLLING_TIME_STEPS, target.state_in: state_train})
    doubleQ = Q2[range(Config.DRQN_BATCH_SIZE * Config.UNROLLING_TIME_STEPS), Q1]
    targetQ = rewards + 0.99 * doubleQ * ~deads
    sess.run(main.optimize, feed_dict={main.input: input_image_drqn(states), main.batch_size: Config.DRQN_BATCH_SIZE, main.unrolling_time_steps: Config.UNROLLING_TIME_STEPS, main.state_in: state_train, main.targetQ: targetQ, main.actions: actions})