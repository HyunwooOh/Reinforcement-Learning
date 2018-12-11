import gym
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from collections import deque
import time
from utils.Config import Config
from utils.valuebase_utils import epsilon_greedy_policy, train_drqn, train_double_drqn
from utils.common_utils import input_image_drqn, pre_proc, get_copy_var_ops_hard
from utils.common_utils import check_life, cal_time, setup_summary

class DRQN():
    def __init__(self, args, action_size, scope):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=Config.ADAM_LEARNING_RATE)
        LSTM = tf.contrib.rnn.BasicLSTMCell(num_units=Config.DRQN_HSIZE)
        with tf.variable_scope(scope):
            self.batch_size = tf.placeholder(tf.int32, ())
            self.unrolling_time_steps = tf.placeholder(tf.int32, ())
            self.input = tf.placeholder(tf.float32, [None, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, 1])
            self.conv1 = layers.conv2d(self.input, 32, 8, 4, 'VALID')
            self.conv2 = layers.conv2d(self.conv1, 64, 4, 2, 'VALID')
            self.conv3 = layers.conv2d(self.conv2, 64, 3, 1, 'VALID')
            self.conv4 = layers.conv2d(self.conv3, Config.DRQN_HSIZE, 7, 1, 'VALID')
            self.conv_flat = layers.flatten(self.conv4)
            self.conv_out = tf.reshape(self.conv_flat, [self.batch_size, self.unrolling_time_steps, Config.DRQN_HSIZE])
            self.state_in = LSTM.zero_state(self.batch_size, tf.float32)
            self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs = self.conv_out, initial_state=self.state_in, cell = LSTM, dtype=tf.float32)
            self.rnn_out = tf.reshape(self.rnn, shape=[-1, Config.DRQN_HSIZE])
            if args.dueling == "True":
                self.streamV, self.streamA = tf.split(self.rnn_out, 2, 1)
                self.AW = tf.Variable(tf.random_normal([Config.DRQN_HSIZE // 2, action_size]))
                self.VW = tf.Variable(tf.random_normal([Config.DRQN_HSIZE // 2, 1]))
                self.advantage = tf.matmul(self.streamA, self.AW)
                self.value = tf.matmul(self.streamV, self.VW)
                self.Qout = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keep_dims=True))
            else:
                self.Qout = layers.fully_connected(self.rnn_out, action_size, activation_fn=None)
            self.targetQ = tf.placeholder(tf.float32, shape=[None])
            self.actions = tf.placeholder(tf.int32, shape=[None])
            self.action_onehot = tf.one_hot(self.actions, action_size, dtype=tf.float32)  # (?, 4)
            self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.action_onehot), axis=1) # (?, )
            self.td_err = tf.square(self.targetQ-self.Q)
            if args.skill == "doom":
                self.maskA = tf.zeros([self.batch_size, self.unrolling_time_steps // 2])
                self.maskB = tf.ones([self.batch_size, self.unrolling_time_steps // 2])
                self.mask = tf.concat([self.maskA, self.maskB], 1)
                self.mask = tf.reshape(self.mask, [-1])
                self.loss = tf.reduce_mean(self.td_err * self.mask)
            else:
                self.loss = tf.reduce_mean(self.td_err)
            self.optimize =self.optimizer.minimize(self.loss)

def train(args):
    env = gym.make(args.game+"-v4")
    ACTION_SIZE = env.action_space.n
    epsilon = 1.0
    start_time = time.time()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    with tf.Session(config=config) as sess:
        ############################
        summary_placeholders, update_ops, summary_op = setup_summary(["Average_Max_Q/Episode", "Total Reward/Episode"])
        summary_writer = tf.summary.FileWriter('summary/'+args.game+"/drqn/", sess.graph)
        ############################
        main = DRQN(args, ACTION_SIZE, "main")
        target = DRQN(args, ACTION_SIZE, "target")
        if args.double == "True": train_model = train_double_drqn
        else: train_model = train_drqn
        sess.run(tf.global_variables_initializer())
        update_target_network = get_copy_var_ops_hard(from_scope="main", to_scope="target")
        sess.run(update_target_network)
        epoch, global_step = 1, 0
        memory = deque(maxlen=1000)
        for episode in range(999999999):
            memory_epi = []
            done, dead = False, False
            step, score, start_life = 0, 0, check_life(env)
            state_in = (np.zeros([1, Config.DRQN_HSIZE]), np.zeros([1, Config.DRQN_HSIZE]))
            avg_q_max = 0
            observe = env.reset()
            for _ in range(random.randint(1, 30)):
                observe, _, _, _ = env.step(1)
            state = np.reshape(pre_proc(observe), [1, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, 1])
            while not done:
                step+=1
                global_step += 1
                ############## choose action ##############
                Q, next_state_in = sess.run([main.Qout, main.rnn_state], feed_dict = {main.input:input_image_drqn(state), main.state_in: state_in, main.batch_size: 1, main.unrolling_time_steps: 1})
                avg_q_max += np.amax(Q, axis = 1)
                action = epsilon_greedy_policy(epsilon, ACTION_SIZE, Q)
                ################ next step ################
                next_observe, reward, done, info = env.step(action)
                ###########################################
                score += reward
                next_state = np.reshape(pre_proc(next_observe), [1, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, 1])
                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']
                reward = np.clip(reward, -1., 1.)
                ############ append experiment ############
                memory_epi.append(np.reshape(np.array([np.copy(state), action, reward, np.copy(next_state), dead]), [1, 5]))
                if dead: dead = False
                else:
                    state = next_state
                    state_in = next_state_in
                ############### train model ###############
                if global_step > Config.TRAIN_START:
                    if epsilon > Config.EPSILON_END: epsilon -= (Config.EPSILON_START-Config.EPSILON_END)/Config.EPSILON_EXPLORATION
                    train_model(sess, main, target, memory)
                    if global_step % Config.TARGET_UPDATE_RATE == 0: sess.run(update_target_network)
                ################ terminated ################
                if done:
                    memory.append(memory_epi)
                    now_time = time.time()
                    hour, min, sec = cal_time(now_time-start_time)
                    print("[%3d : %2d : %5.2f] Episode: %7d | Score: %4d | Avg_max_Qvalue: %.4f | Global_step: %d"%(hour, min, sec, episode, score, avg_q_max/step, global_step))
                    f = open(args.report_path + args.report_file_name, 'a')
                    f.write("%f\t%d\t%d\t%d\t%d\n" % (now_time - start_time, episode, score, global_step, step))
                    f.close()
                    summary_stats = [avg_q_max/step, score]#, step]
                    for i in range(len(summary_stats)):
                        sess.run(update_ops[i], feed_dict={summary_placeholders[i]: float(summary_stats[i])})
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, episode + 1)