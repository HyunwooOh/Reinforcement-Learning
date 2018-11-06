import gym
import random
import threading
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import time
from utils.common_utils import input_image, pre_proc, get_copy_var_ops_hard
from utils.common_utils import check_life, cal_time, setup_summary, total_parameters
from utils.policybase_utils import train_global

global episode, global_step
episode = 0
global_step = 0
EPISODES = 8000000

class ActorCritic():
    def __init__(self, action_size, num_frame, scope):
        print("make one loss", scope)
        self.optimizer = tf.train.RMSPropOptimizer(2.5e-4, decay=0.99, epsilon=0.01)
        self.action_size = action_size
        with tf.variable_scope(scope):
            self.batch_size = tf.placeholder(tf.float32, ())
            self.input = tf.placeholder(tf.float32, [None, 84, 84, num_frame])
            self.conv1 = layers.conv2d(self.input, 16, 8, 4, padding='VALID')
            self.conv2 = layers.conv2d(self.conv1, 32, 4, 2, padding='VALID')
            self.conv_flat = layers.flatten(self.conv2)
            self.fc = layers.fully_connected(self.conv_flat, 256)
            self.policy = layers.fully_connected(self.fc, self.action_size, activation_fn=tf.nn.softmax)
            self.value = layers.fully_connected(self.fc, 1, activation_fn=None)
            if scope != 'global':
                self.actions = tf.placeholder(tf.int32, [None, ])
                self.action_onehot = tf.one_hot(self.actions, self.action_size, axis=1)  # (?, 3)
                self.discounted_R = tf.placeholder(tf.float32, [None, 1])

                self.critic_loss = tf.square(self.discounted_R - self.value) #(?, 1)
                self.critic_loss = tf.reduce_sum(self.critic_loss) #()

                self.selected_action_prob = tf.reduce_sum(self.policy*self.action_onehot, axis=1, keep_dims = True) #(?, 1)
                self.log_pi = tf.log(tf.maximum(self.selected_action_prob, 1e-10))  # (?, 1)
                self.actor_loss = self.log_pi * (self.discounted_R - tf.stop_gradient(self.value)) #(?, 1)
                self.actor_loss = tf.reduce_sum(self.actor_loss) #()
                self.entropy = tf.reduce_sum(self.policy * tf.log(tf.maximum(self.policy, 1e-10)), axis=1, keep_dims=True) #(?, 1)
                self.entropy = tf.reduce_sum(self.entropy) #()

                self.loss = 0.5 * self.critic_loss - self.actor_loss + 0.01 * self.entropy #()
                #self.loss = tf.reduce_sum(self.loss_)

                self.local_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.local_gradients = tf.gradients(self.loss, self.local_params)
                self.local_gradients, _ = tf.clip_by_global_norm(self.local_gradients, 40.0)
                self.global_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = self.optimizer.apply_gradients(zip(self.local_gradients, self.global_params))
                self.optimize = self.apply_grads

class ActorCritic2():
    def __init__(self, action_size, num_frame, scope):
        print("make two loss", scope)
        self.optimizer = tf.train.RMSPropOptimizer(2.5e-4, decay=0.99, epsilon=0.01)
        self.action_size = action_size
        with tf.variable_scope(scope):
            self.input = tf.placeholder(tf.float32, [None, 84, 84, num_frame])
            with tf.variable_scope("actor"):
                self.conv1 = layers.conv2d(self.input, 16, 8, 4, padding='VALID')
                self.conv2 = layers.conv2d(self.conv1, 32, 4, 2, padding='VALID')
                self.conv_flat = layers.flatten(self.conv2)
                self.fc = layers.fully_connected(self.conv_flat, 256)
                self.policy = layers.fully_connected(self.fc, self.action_size, activation_fn=tf.nn.softmax)
            with tf.variable_scope("critic"):
                self.conv1 = layers.conv2d(self.input, 16, 8, 4, padding='VALID')
                self.conv2 = layers.conv2d(self.conv1, 32, 4, 2, padding='VALID')
                self.conv_flat = layers.flatten(self.conv2)
                self.fc = layers.fully_connected(self.conv_flat, 256)
                self.value = layers.fully_connected(self.fc, 1, activation_fn=None)
            if scope != 'global':
                self.actions = tf.placeholder(tf.int32, [None, ])
                self.action_onehot = tf.one_hot(self.actions, self.action_size, axis=1)  # (?, 3)
                self.discounted_R = tf.placeholder(tf.float32, [None, 1])

                self.critic_loss = tf.square(self.discounted_R - self.value) #(?, 1)
                self.critic_loss = tf.reduce_sum(self.critic_loss)
                self.selected_action_prob = tf.reduce_sum(self.policy*self.action_onehot, axis=1, keep_dims = True) #(?, 1)
                self.log_pi = tf.log(tf.maximum(self.selected_action_prob, 1e-10))  # (?, 1)
                self.actor_loss_ = self.log_pi * (self.discounted_R - tf.stop_gradient(self.value)) #(?, 1)
                self.entropy = tf.reduce_sum(self.policy * tf.log(tf.maximum(self.policy, 1e-10)), axis=1, keep_dims=True) #(?, 1)
                self.actor_loss = tf.reduce_sum(-self.actor_loss_ + 0.01 * self.entropy)

                self.local_critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope+"/critic")
                self.local_critic_grad = tf.gradients(self.critic_loss, self.local_critic_params)
                self.local_critic_grad, _ = tf.clip_by_global_norm(self.local_critic_grad, 40.0)
                self.global_critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "global/critic")
                self.apply_critic_grads = self.optimizer.apply_gradients(zip(self.local_critic_grad, self.global_critic_params))

                self.local_actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope+"/actor")
                self.local_actor_grad = tf.gradients(self.actor_loss, self.local_actor_params)
                self.local_actor_grad, _ = tf.clip_by_global_norm(self.local_actor_grad, 40.0)
                self.global_actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "global/actor")
                self.apply_actor_grads = self.optimizer.apply_gradients(zip(self.local_actor_grad, self.global_actor_params))

                self.optimize = [self.apply_critic_grads, self.apply_actor_grads]


class Worker(threading.Thread):
    def __init__(self, name, sess, action_size, num_frame, game_name, ACNet, start_time, summary_ops):
        threading.Thread.__init__(self)
        self.name = name
        self.sess = sess
        self.action_size = action_size
        self.num_frame = num_frame
        self.game_name = game_name
        self.local_AC = ACNet(action_size, num_frame, name)
        self.start_time = start_time
        [self.summary_op, self.summary_placeholders, self.update_ops, self.summary_writer] = summary_ops
        self.update_local_network = get_copy_var_ops_hard(from_scope="global", to_scope=name)
        self.histories, self.actions, self.rewards = [], [], []
        self.avg_p_max = 0
        self.t_max = 20
        self.t = 0

    def run(self):
        global episode
        global global_step
        env = gym.make(self.game_name)
        while episode < EPISODES:
            done, dead = False, False
            step, score, start_life = 0, 0, check_life(env)
            observe = env.reset()
            for _ in range(random.randint(1, 30)):
                observe, _, _, _ = env.step(1)
            state = pre_proc(observe) # state shape: (84*84*1)
            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape(history, (1, 84, 84, self.num_frame)) # use history as input
            while not done:
                step+=1
                self.t += 1
                ############## choose action ##############
                policy = self.sess.run(self.local_AC.policy, feed_dict = {self.local_AC.input:input_image(history, self.num_frame)})[0]
                self.avg_p_max += np.amax(policy)
                action = np.random.choice(self.action_size, 1, p=policy)[0]
                ############### for breakout ###############
                if self.game_name == "BreakoutDeterministic-v4":
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
                next_state = np.reshape(pre_proc(next_observe), (1, 84, 84, 1))
                next_history = np.append(next_state, history[:, :, :, :(self.num_frame-1)], axis=3)
                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']
                score += reward
                ############ append experiment ############
                self.append_sample(np.copy(history), action, reward)
                if not dead: history = next_history
                ############### train model ###############
                if self.t >= self.t_max or done:
                    train_global(self.sess, self.local_AC, self.histories, self.actions, self.rewards, done, self.num_frame) # Train global network
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
                    f = open("./report/a3c_result.txt", 'a')
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
    if args.num_loss == 1:    A3C_Net = ActorCritic
    else: A3C_Net = ActorCritic2
    if args.game == "BreakoutDeterministic": action_size = 3
    sess = tf.Session()
    summary_placeholders, update_ops, summary_op = setup_summary(["Total Reward/Episode", "Average_Max_Prob/Episode"])
    summary_writer = tf.summary.FileWriter('summary/a3c/', sess.graph)
    summary_ops = [summary_op, summary_placeholders, update_ops, summary_writer]
    with tf.device("/cpu:0"):
        global_AC = A3C_Net(action_size, args.num_frame, "global")
        workers = [Worker("Worker_%i"%i, sess, action_size, args.num_frame, args.game+"-v4", A3C_Net, start_time, summary_ops) for i in range(args.num_cpu)]
    sess.run(tf.global_variables_initializer())
    for worker in workers:
        time.sleep(1)
        worker.start()