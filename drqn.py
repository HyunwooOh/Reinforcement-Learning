import tensorflow as tf
import numpy as np
import gym
import random
import sys
from utils.valuebase_utils import train_drqn, train_double_drqn, experience_buffer_drqn
from utils.common_utils import input_image, pre_proc_scalar, get_copy_var_ops_hard, get_copy_var_ops_soft
from utils.common_utils import check_life, cal_time, setup_summary, make_path, total_parameters

class DRQN():
    def __init__(self, args, h_size, action_size, rnn_cell, myScope):
        self.scalarInput = tf.placeholder(shape=[None, 84*84], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 84, 84, 1])
        self.conv1 = tf.contrib.layers.convolution2d(inputs=self.imageIn, num_outputs=32, kernel_size=[8, 8], stride=[4, 4],
                                              padding='VALID', scope=myScope + '_conv1')
        self.conv2 = tf.contrib.layers.convolution2d(inputs=self.conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2],
                                              padding='VALID', scope=myScope + '_conv2')
        self.conv3 = tf.contrib.layers.convolution2d(inputs=self.conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1],
                                              padding='VALID', scope=myScope + '_conv3')
        self.conv4 = tf.contrib.layers.convolution2d(inputs=self.conv3, num_outputs=h_size, kernel_size=7, stride=1, padding='VALID', scope=myScope+'_conv4')
        self.unrolling_time_steps = tf.placeholder(dtype=tf.int32, shape=[])
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
        self.convFlat = tf.reshape(tf.contrib.layers.flatten(self.conv4), [self.batch_size, self.unrolling_time_steps, h_size])
        self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs = self.convFlat, initial_state=self.state_in, cell=rnn_cell, dtype=tf.float32, scope=myScope+'_lstm')

        self.rnn = tf.reshape(self.rnn, shape=[-1, h_size])
        self.Qout = tf.contrib.layers.fully_connected(inputs=self.rnn, num_outputs=action_size, activation_fn=None)
        self.predict = tf.argmax(self.Qout, 1)
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, action_size, dtype=tf.float32)
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        self.td_error = tf.square(self.targetQ - self.Q)

        if args.skill == "doom":
            self.maskA = tf.zeros([self.batch_size, self.unrolling_time_steps // 2])
            self.maskB = tf.ones([self.batch_size, self.unrolling_time_steps // 2])
            self.mask = tf.concat([self.maskA, self.maskB], 1)
            self.mask = tf.reshape(self.mask, [-1])
            self.loss = tf.reduce_mean(self.td_error * self.mask)
        else:
            self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

def get_q(sess, main, state, state_in):
    q = sess.run(main.Qout, feed_dict={main.scalarInput: [state / 255.0], main.unrolling_time_steps: 1,
                                            main.state_in: state_in, main.batch_size: 1})
    return np.amax(q, axis=1)

def get_action_drqn(sess, epsilon, main, state, state_in, action_size, global_steps):
    if np.random.rand(1) < epsilon or global_steps < 10000:
        next_state_in = sess.run(main.rnn_state,
                                 feed_dict={main.scalarInput: [state / 255.0], main.unrolling_time_steps: 1,
                                            main.state_in: state_in, main.batch_size: 1})
        action = np.random.randint(0, action_size)
    else:
        a, next_state_in = sess.run([main.predict, main.rnn_state],
                                    feed_dict={main.scalarInput: [state / 255.0], main.unrolling_time_steps: 1,
                                               main.state_in: state_in, main.batch_size: 1})
        action = a[0]
    return next_state_in, action

def train(args):
    print("Opening environment...")
    env = gym.make(args.game+"-v4")
    ACTION_SIZE = env.action_space.n
    ENV_LIFE = check_life(env)
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=args.h_size, state_is_tuple=True)
    cellT = tf.contrib.rnn.BasicLSTMCell(num_units=args.h_size, state_is_tuple=True)
    main = DRQN(args, args.h_size, ACTION_SIZE, cell, 'main')
    target = DRQN(args, args.h_size, ACTION_SIZE, cellT, 'target')
    if args.double == "True": train_model = train_double_drqn
    else: train_model = train_drqn
    targetOps = get_copy_var_ops_soft(tf.trainable_variables(), 0.001)
    epsilon = args.epsilon_start
    epsilon_drop = (args.epsilon_start - args.epsilon_end) / args.epsilon_exploration

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    with tf.Session(config=config) as sess:
        total_parameters()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=1)
        for op in targetOps: sess.run(op)
        myBuffer = experience_buffer_drqn()
        global_steps=0
        for episode in range(1, 99999999):
            episode_buffer = []
            done = False
            dead = False
            step, score, start_life = 0, 0, ENV_LIFE
            state_in = (np.zeros([1, args.h_size]), np.zeros([1, args.h_size]))

            observe = env.reset()
            for _ in range(random.randint(1, 30)):
                observe, _, _, _ = env.step(1)
            state = pre_proc_scalar(observe)
            sum_q_max_epi=0
            while not done:
                step+=1
                global_steps+=1
                sum_q_max_epi += get_q(sess, main, state, state_in)
                next_state_in, action = get_action_drqn(sess, epsilon, main, state, state_in, ACTION_SIZE, global_steps)
                next_observe, reward, done, info = env.step(action)
                next_state = pre_proc_scalar(next_observe)
                score += reward
                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']
                reward = np.clip(reward, -1., 1.)
                episode_buffer.append(np.reshape(np.array([state, action, reward, next_state, dead]), [1, 5]))
                if dead: dead = False
                else:
                    state = next_state
                    state_in = next_state_in
                if done:
                    print("Episode: %d | Score: %d | Avg_Q_max: %.8f | Global_steps: %d" % (episode, score, sum_q_max_epi/step, global_steps))
                    if args.report == 'True':
                        f_epi = open(args.report_path + args.report_file_name, 'a')
                        f_epi.write("%d\t%d\t%d\t%.8f\n" % (episode, global_steps, score, sum_q_max_epi/step))
                        f_epi.close()
                if global_steps > args.train_start:
                    if epsilon > args.epsilon_end: epsilon -= epsilon_drop
                    if global_steps % args.target_update_rate:
                        for op in targetOps: sess.run(op)
                    train_model(sess, args, main, target, myBuffer)
            bufferArray = np.array(episode_buffer)
            episode_buffer = list(zip(bufferArray))
            myBuffer.add(episode_buffer)
            if global_steps > args.train_end: sys.exit()
