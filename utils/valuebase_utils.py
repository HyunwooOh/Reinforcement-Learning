import numpy as np
import random
from utils.common_utils import input_image

##############################
########### POLICY ###########
##############################
def epsilon_greedy_policy(epsilon, ACTION_SIZE, Q):
    if epsilon > np.random.rand(1):
        action = np.random.randint(ACTION_SIZE)
    else:
        action = np.argmax(Q)
    return action

def train_async_dqn(sess, num_frame, local_AC, target, memory):
    minibatch = memory
    histories = np.vstack([x[0][:, :, :, :num_frame] for x in minibatch])
    actions = np.array([x[1] for x in minibatch])
    rewards = np.array([x[2] for x in minibatch])
    next_histories = np.vstack([x[0][:, :, :, 1:] for x in minibatch])
    deads = np.array([x[3] for x in minibatch])
    targetQ = rewards + 0.99 * np.max(sess.run(target.Qout, feed_dict={target.input:input_image(next_histories, num_frame), target.batch_size:len(minibatch)}), axis=1) * ~deads
    sess.run(local_AC.optimize, feed_dict={local_AC.input:input_image(histories, num_frame), local_AC.batch_size:len(minibatch), local_AC.targetQ:targetQ, local_AC.actions:actions})

###############################
########### For DQN ###########
###############################
def train_dqn(args, sess, main, target, memory):
    minibatch = random.sample(memory, args.batch_size)
    histories = np.vstack([x[0][:, :, :, :args.num_frame] for x in minibatch])
    actions = np.array([x[1] for x in minibatch])
    rewards = np.array([x[2] for x in minibatch])
    next_histories = np.vstack([x[0][:, :, :, 1:] for x in minibatch])
    deads = np.array([x[3] for x in minibatch])
    targetQ = rewards + 0.99 * np.max(sess.run(target.Qout, feed_dict={target.input:input_image(next_histories, args.num_frame), target.batch_size:args.batch_size}), axis=1) * ~deads
    sess.run(main.optimize, feed_dict={main.input:input_image(histories, args.num_frame), main.batch_size:args.batch_size, main.targetQ:targetQ, main.actions:actions})

def train_double_dqn(args, sess, main, target, memory):
    minibatch = random.sample(memory, args.batch_size)
    histories = np.vstack([x[0][:, :, :, :args.num_frame] for x in minibatch])
    actions = np.array([x[1] for x in minibatch])
    rewards = np.array([x[2] for x in minibatch])
    next_histories = np.vstack([x[0][:, :, :, 1:] for x in minibatch])
    deads = np.array([x[3] for x in minibatch])
    Q1 = np.argmax(sess.run(main.Qout, feed_dict={main.input:input_image(next_histories, args.num_frame), main.batch_size:args.batch_size}), 1)
    Q2 = sess.run(target.Qout, feed_dict={target.input:input_image(next_histories, args.num_frame), target.batch_size:args.batch_size})
    doubleQ = Q2[range(args.batch_size), Q1]
    targetQ = rewards + 0.99 * doubleQ * ~deads
    sess.run(main.optimize, feed_dict={main.input:input_image(histories, args.num_frame), main.batch_size:args.batch_size, main.targetQ:targetQ, main.actions:actions})

################################
########### For DRQN ###########
################################
class experience_buffer_drqn():
    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self, batch_size, unrolling_time_steps):
        sampled_episodes = random.sample(self.buffer, batch_size)
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - unrolling_time_steps)
            sampledTraces.append(episode[point:point + unrolling_time_steps])
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces, [batch_size * unrolling_time_steps, 5])

def train_drqn(sess, args, main, target, myBuffer):
    state_train = (np.zeros([args.batch_size, args.h_size]), np.zeros([args.batch_size, args.h_size]))
    mini_batch = myBuffer.sample(args.batch_size, args.unrolling_time_steps)
    Q2 = sess.run(target.Qout, feed_dict={target.scalarInput: np.vstack(mini_batch[:, 3] / 255.0),
                                          target.unrolling_time_steps: args.unrolling_time_steps,
                                          target.state_in: state_train, target.batch_size: args.batch_size})
    end_multiplier = -(mini_batch[:, 4] - 1)
    targetQ = mini_batch[:, 2] + (0.99 * np.max(Q2, axis=1) * end_multiplier)

    sess.run(main.updateModel, feed_dict={main.scalarInput: np.vstack(mini_batch[:, 0] / 255.0), main.targetQ: targetQ,
                                          main.actions: mini_batch[:, 1],
                                          main.unrolling_time_steps: args.unrolling_time_steps,
                                          main.state_in: state_train, main.batch_size: args.batch_size})

def train_double_drqn(sess, args, main, target, myBuffer):
    state_train = (np.zeros([args.batch_size, args.h_size]), np.zeros([args.batch_size, args.h_size]))
    mini_batch = myBuffer.sample(args.batch_size, args.unrolling_time_steps)
    Q1 = sess.run(main.predict, feed_dict={main.scalarInput: np.vstack(mini_batch[:, 3] / 255.0),
                                           main.unrolling_time_steps: args.unrolling_time_steps,
                                           main.state_in: state_train, main.batch_size: args.batch_size})
    Q2 = sess.run(target.Qout, feed_dict={target.scalarInput: np.vstack(mini_batch[:, 3] / 255.0),
                                          target.unrolling_time_steps: args.unrolling_time_steps,
                                          target.state_in: state_train, target.batch_size: args.batch_size})
    end_multiplier = -(mini_batch[:, 4] - 1)
    doubleQ = Q2[range(args.batch_size * args.unrolling_time_steps), Q1]
    targetQ = mini_batch[:, 2] + (0.99 * doubleQ * end_multiplier)

    sess.run(main.updateModel, feed_dict={main.scalarInput: np.vstack(mini_batch[:, 0] / 255.0), main.targetQ: targetQ,
                                          main.actions: mini_batch[:, 1],
                                          main.unrolling_time_steps: args.unrolling_time_steps,
                                          main.state_in: state_train, main.batch_size: args.batch_size})
