import numpy as np
from utils.Config import Config
from utils.common_utils import input_image

def train_global(sess, local_AC, histories, actions, rewards, last_history, dead):
    histories = np.vstack([x for x in histories])
    discounted_R = np.zeros_like(rewards) #(?, )
    if dead: running_add = 0
    else: running_add = sess.run(local_AC.value, feed_dict={local_AC.batch_size:1, local_AC.input:input_image(last_history)})
    for i in reversed(range(0, len(rewards))):
        running_add = rewards[i] + running_add * Config.DISCOUNT_FACTOR
        discounted_R[i] = running_add
    feed_dict = {local_AC.input:input_image(histories), local_AC.actions:actions, local_AC.discounted_R:discounted_R, local_AC.batch_size:len(discounted_R)}
    sess.run(local_AC.optimize, feed_dict=feed_dict)
