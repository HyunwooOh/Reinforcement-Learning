import gym
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from collections import deque
import time
from utils.Config import Config
from utils.valuebase_utils import epsilon_greedy_policy, train_dqn, train_double_dqn
from utils.common_utils import input_image, pre_proc, get_copy_var_ops_hard
from utils.common_utils import check_life, cal_time, setup_summary

class DQN():
    def __init__(self, args, action_size, scope):
        #self.optimizer = tf.train.RMSPropOptimizer(2.5e-4, decay=0.99, epsilon=0.01)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=Config.ADAM_LEARNING_RATE)
        with tf.variable_scope(scope):
            self.batch_size = tf.placeholder(tf.float32, ())
            self.input = tf.placeholder(tf.float32, [None, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, Config.NUM_FRAME])
            self.conv1 = layers.conv2d(self.input, 32, 8, 4, 'VALID')
            self.conv2 = layers.conv2d(self.conv1, 64, 4, 2, 'VALID')
            self.conv3 = layers.conv2d(self.conv2, 64, 3, 1, 'VALID')
            self.conv_flat = layers.flatten(self.conv3)
            if args.dueling == 'True':
                self.streamV, self.streamA = tf.split(self.conv_flat, 2, 1)
                self.value = layers.fully_connected(self.streamV, 1)
                self.advantage = layers.fully_connected(self.streamA, action_size)
                self.Qout = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keep_dims=True))
            else:
                self.fc = layers.fully_connected(self.conv_flat, 512)
                self.Qout = tf.contrib.layers.fully_connected(self.fc, action_size, activation_fn=None) # (?, 4)
            self.targetQ = tf.placeholder(tf.float32, shape=[None])
            self.actions = tf.placeholder(tf.int32, shape=[None])
            self.action_onehot = tf.one_hot(self.actions, action_size, dtype=tf.float32)  # (?, 4)
            self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.action_onehot), axis=1) # (?, )
            self.td_err = tf.square(self.targetQ-self.Q)
            self.loss = tf.reduce_mean(self.td_err)
            self.trainModel =self.optimizer.minimize(self.loss)
            self.optimize = self.trainModel

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
        summary_writer = tf.summary.FileWriter('summary/dqn/'+args.game+"/", sess.graph)
        ############################
        main = DQN(args, ACTION_SIZE, "main")
        target = DQN(args, ACTION_SIZE, "target")
        if args.double == "True": train_model = train_double_dqn
        else: train_model = train_dqn
        sess.run(tf.global_variables_initializer())
        update_target_network = get_copy_var_ops_hard(from_scope="main", to_scope="target")
        sess.run(update_target_network)
        epoch, global_step = 1, 0
        memory = deque(maxlen=Config.MEMORY_SIZE)
        for episode in range(999999999):
            done, dead = False, False
            step, score, start_life = 0, 0, check_life(env)
            avg_q_max = 0

            observe = env.reset()
            for _ in range(random.randint(1, 30)):
                observe, _, _, _ = env.step(1)
            state = np.reshape(pre_proc(observe), [1, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, 1])
            history = state
            for _ in range(Config.NUM_FRAME):
                history = np.concatenate((history, state), axis=3)
            while not done:
                step+=1
                global_step += 1
                ############## choose action ##############
                Q = sess.run(main.Qout, feed_dict = {main.input:input_image(history[:,:,:,:Config.NUM_FRAME])})
                avg_q_max += np.amax(Q, axis = 1)
                action = epsilon_greedy_policy(epsilon, ACTION_SIZE, Q)
                ################ next step ################
                next_observe, reward, done, info = env.step(action)
                ###########################################
                score += reward
                history[:, :, :, Config.NUM_FRAME] = pre_proc(next_observe)
                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']
                reward = np.clip(reward, -1., 1.)
                ############ append experiment ############
                memory.append((np.copy(history), action, reward, dead))
                if dead: dead = False
                else: history[:, :, :, :Config.NUM_FRAME] = history[:, :, :, 1:]
                ############### train model ###############
                if global_step > Config.TRAIN_START:
                    if epsilon > Config.EPSILON_END: epsilon -= (Config.EPSILON_START-Config.EPSILON_END)/Config.EPSILON_EXPLORATION
                    train_model(sess, main, target, memory)
                    if global_step % Config.TARGET_UPDATE_RATE == 0: sess.run(update_target_network)
                ################ terminated ################
                if done:
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