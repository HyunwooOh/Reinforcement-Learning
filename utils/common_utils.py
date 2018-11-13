import numpy as np
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.transform import resize
import os
from utils.Config import Config

def tf_log(X):
    return tf.log(tf.maximum(X, Config.LOG_EPSILON))

def input_image(X):
    return np.reshape(np.float32(X/255.), [-1, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, Config.NUM_FRAME])

def check_life(env):
    env.reset()
    _, _, _, info = env.step(0)
    env.reset()
    return int(info['ale.lives'])

def pre_proc(X):
    x = np.uint8(resize(rgb2gray(X), (Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH), mode='constant') * 255)
    return x

def pre_proc_scalar(X):
    x = np.uint8(resize(rgb2gray(X), (Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH), mode='constant') * 255)
    return np.reshape(x, [Config.IMAGE_HEIGHT*Config.IMAGE_WIDTH])

def get_copy_var_ops_hard(*, from_scope, to_scope):
    op_holder = []
    from_vars = tf.trainable_variables(from_scope)
    to_vars = tf.trainable_variables(to_scope)
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var.value()))
    return op_holder

def get_copy_var_ops_soft(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def cal_time(sec_):
    sec = sec_%60
    min_ = sec_//60
    min = min_%60
    hour = min_//60
    return hour, min, sec

def setup_summary(list):
    variables = []
    for i in range(len(list)):
        variables.append(tf.Variable(0.))
        tf.summary.scalar(list[i], variables[i])
    summary_vars = [x for x in variables]
    summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
    update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
    summary_op = tf.summary.merge_all()
    return summary_placeholders, update_ops, summary_op

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def total_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        #print(shape)
        #print(len(shape))
        variable_parameters = 1
        for dim in shape:
            #print(dim)
            variable_parameters *= dim.value
        #print(variable_parameters)
        total_parameters += variable_parameters
    print("number of trainable parameters: %d"%(total_parameters))