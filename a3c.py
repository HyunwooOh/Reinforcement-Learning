import gym
import random
import threading
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import time
from utils.Config import Config
from utils.policybase_utils import train_global
from utils.common_utils import pre_proc, get_copy_var_ops_hard, input_image
from utils.common_utils import check_life, cal_time, setup_summary, total_parameters

global episode, global_step
episode = 0
global_step = 0
EPISODES = 8000000

class ActorCritic():
    def __init__(self, action_size, scope):
        print("make one loss", scope)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=Config.ADAM_LEARNING_RATE)
        #self.optimizer = tf.train.RMSPropOptimizer(2.5e-4, decay=0.99, epsilon=0.01)
        self.action_size = action_size
        with tf.variable_scope(scope):
            self.batch_size = tf.placeholder(tf.float32, ())
            self.input = tf.placeholder(tf.float32, [None, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, Config.NUM_FRAME])
            self.conv1 = layers.conv2d(self.input, 16, 8, 4, padding='VALID')
            self.conv2 = layers.conv2d(self.conv1, 32, 4, 2, padding='VALID')
            self.conv_flat = layers.flatten(self.conv2)
            self.fc = layers.fully_connected(self.conv_flat, 256)
            self.policy = layers.fully_connected(self.fc, self.action_size, activation_fn=tf.nn.softmax)
            self.value_ = layers.fully_connected(self.fc, 1, activation_fn=None)
            self.value = tf.reshape(self.value_, [-1])
            if scope != 'global':
                self.actions = tf.placeholder(tf.int32, [None])
                self.action_onehot = tf.one_hot(self.actions, self.action_size, axis=1)  # (?, 3)
                self.discounted_R = tf.placeholder(tf.float32, [None])

                self.critic_loss = tf.reduce_sum(tf.square(self.discounted_R - self.value)) #(?) -> ()
                self.selected_action_prob = tf.reduce_sum(self.policy*self.action_onehot, axis=1) #(?)
                self.log_pi = tf.log(tf.maximum(self.selected_action_prob, Config.LOG_EPSILON))  # (?)
                self.actor_loss = tf.reduce_sum(self.log_pi * (self.discounted_R - tf.stop_gradient(self.value))) #(?) -> ()
                self.entropy = tf.reduce_sum(tf.reduce_sum(self.policy * tf.log(tf.maximum(self.policy, Config.LOG_EPSILON)), axis=1)) #(?, 3) -> (?) -> ()

                self.loss = 0.5 * self.critic_loss - self.actor_loss + Config.ENTROPY_BETA * self.entropy #()

                self.local_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.local_gradients = tf.gradients(self.loss, self.local_params)
                self.local_gradients, _ = tf.clip_by_global_norm(self.local_gradients, Config.GRAD_CLIP_NORM)
                self.global_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = self.optimizer.apply_gradients(zip(self.local_gradients, self.global_params))
                self.optimize = self.apply_grads

class Worker(threading.Thread):
    def __init__(self, name, sess, action_size, game, ActorCritic, start_time, summary_ops, report_name):
        threading.Thread.__init__(self)
        self.name = name
        self.sess = sess
        self.action_size = action_size
        self.game = game
        self.local_AC = ActorCritic(action_size, name)
        self.start_time = start_time
        [self.summary_op, self.summary_placeholders, self.update_ops, self.summary_writer] = summary_ops
        self.report_name = report_name
        self.update_local_network = get_copy_var_ops_hard(from_scope="global", to_scope=name)
        self.histories, self.actions, self.rewards= [], [], []
        self.avg_p_max = 0
        self.t_max = Config.T_MAX
        self.t = 0

    def run(self):
        global episode
        global global_step
        env = gym.make(self.game+'-v4')
        while episode < EPISODES:
            done, dead = False, False
            step, score, start_life = 0, 0, check_life(env)
            observe = env.reset()
            for _ in range(random.randint(1, 30)):
                observe, _, _, _ = env.step(1)
            state = pre_proc(observe) # state shape: (84*84*1)
            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape(history, (1, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, Config.NUM_FRAME)) # use history as input
            while not done:
                step+=1
                self.t += 1
                ############## choose action ##############
                policy = self.sess.run(self.local_AC.policy, feed_dict = {self.local_AC.input:input_image(history)})[0]
                self.avg_p_max += np.amax(policy)
                action = np.random.choice(self.action_size, 1, p=policy)[0]
                ############### for breakout ###############
                if self.game == "BreakoutDeterministic":
                    if action == 0: real_action = 1
                    elif action == 1: real_action = 2
                    else: real_action = 3
                    if dead:
                        action = 0
                        real_action = 1
                        dead = False
                else:
                    real_action = action
                    if dead: dead = False
                ################ next step ################
                next_observe, reward, done, info = env.step(real_action)
                ###########################################
                next_state = np.reshape(pre_proc(next_observe), (1, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, 1))
                next_history = np.append(next_state, history[:, :, :, :(Config.NUM_FRAME-1)], axis=3)
                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']
                score += reward
                ############ append experiment ############
                self.append_sample(np.copy(history), action, reward)
                if dead:
                    history = np.stack((next_state, next_state, next_state, next_state), axis=2)
                    history = np.reshape([history], (1, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, Config.NUM_FRAME))
                    last_history = np.copy(history)
                else:
                    history = next_history
                    last_history = np.copy(history)
                ############### train model ###############
                if self.t >= self.t_max or dead:
                    train_global(self.sess, self.local_AC, self.histories, self.actions, self.rewards, last_history, dead) # Train global network
                    self.histories, self.actions, self.rewards = [], [], []
                    self.sess.run(self.update_local_network) # Synchronize thread-specific parameters
                    self.t = 0
                ################ terminated ################
                if done:
                    episode += 1
                    global_step += step
                    now_time = time.time()
                    hour, min, sec = cal_time(now_time - self.start_time)
                    print("[%3d : %2d : %5.2f] Episode: %7d | Score: %4d | Avg_max_Policy: %.4f | Global_step: %d"%(hour, min, sec, episode, score, self.avg_p_max/step, global_step))
                    f = open(self.report_name, 'a')
                    f.write("%f\t%d\t%d\t%d\t%f\n" % (now_time - self.start_time, episode, global_step, score, self.avg_p_max/step))
                    f.close()
                    stats = [score, self.avg_p_max/step]
                    for i in range(len(stats)):
                        self.sess.run(self.update_ops[i], feed_dict={self.summary_placeholders[i]:float(stats[i])})
                    summary_str = self.sess.run(self.summary_op)
                    self.summary_writer.add_summary(summary_str, episode + 1)
                    self.avg_p_max = 0

    def append_sample(self, history, action, reward):
        self.histories.append(history)
        self.actions.append(action)
        self.rewards.append(reward)

def train(args):
    print("A3C")
    env = gym.make(args.game+"-v4")
    env.reset()
    start_time = time.time()
    action_size = env.action_space.n
    if args.game == "BreakoutDeterministic": action_size = 3
    sess = tf.Session()
    summary_placeholders, update_ops, summary_op = setup_summary(["Total Reward/Episode", "Average_Max_Prob/Episode"])
    summary_writer = tf.summary.FileWriter('summary/a3c/'+args.game+'/', sess.graph)
    summary_ops = [summary_op, summary_placeholders, update_ops, summary_writer]
    with tf.device("/cpu:0"):
        global_AC = ActorCritic(action_size, "global")
        workers = [Worker("Worker_%i"%i, sess, action_size, args.game, ActorCritic, start_time, summary_ops, args.report_path+args.report_file_name) for i in range(args.num_cpu)]
    sess.run(tf.global_variables_initializer())
    for worker in workers:
        time.sleep(1)
        worker.start()