import numpy as np
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.transform import resize
import os

def input_image(X, num_frame):
    return np.reshape(np.float32(X/255.), [-1, 84, 84, num_frame])

def check_life(env):
    env.reset()
    _, _, _, info = env.step(0)
    env.reset()
    return int(info['ale.lives'])

def pre_proc(X):
    x = np.uint8(resize(rgb2gray(X), (84, 84), mode='constant') * 255)
    return x

def pre_proc_scalar(X):
    x = np.uint8(resize(rgb2gray(X), (84, 84), mode='constant') * 255)
    return np.reshape(x, [84*84])

def get_copy_var_ops_hard2(*, from_scope, to_scope):
    op_holder = []
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=to_scope)
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var.value()))
    return op_holder
def get_copy_var_ops_hard(from_scope, to_scope):
    """
    Create operations to mirror the values from all trainable variables in from_scope to to_scope.
    """
    from_tvs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=from_scope)
    to_tvs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=to_scope)

    from_dict = {var.name: var for var in from_tvs}
    to_dict = {var.name: var for var in to_tvs}
    copy_ops = []
    for to_name, to_var in to_dict.items():
        from_name = to_name.replace(to_scope, from_scope)
        from_var = from_dict[from_name]
        op = to_var.assign(from_var.value())
        copy_ops.append(op)

    return copy_ops
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