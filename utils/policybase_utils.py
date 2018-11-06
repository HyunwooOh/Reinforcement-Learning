import numpy as np
from utils.common_utils import input_image

def train_global(sess, local_AC, histories, actions, rewards, done, num_frame):
    histories = np.vstack([x for x in histories])
    discounted_R = np.zeros_like(rewards) #(?, )
    if done: running_add = 0
    else: running_add = sess.run(local_AC.value, feed_dict={local_AC.input:input_image(histories[-1], num_frame)})[0]
    for i in reversed(range(0, len(rewards))):
        running_add = rewards[i] + running_add * 0.99
        discounted_R[i] = running_add
    discounted_R = np.reshape(discounted_R, [len(discounted_R), 1])
    feed_dict = {local_AC.input:input_image(histories, num_frame), local_AC.actions:actions, local_AC.discounted_R:discounted_R, local_AC.batch_size:len(discounted_R)}
    sess.run(local_AC.optimize, feed_dict=feed_dict)