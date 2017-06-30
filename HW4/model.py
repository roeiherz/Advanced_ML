import tensorflow as tf
import numpy as np

# Define layers size
from HW4.train import NOF_BATCHES_PER_EPOCH

INPUT_SIZE = 8
H1_SIZE = 30
H2_SIZE = 30
OUPUT_SIZE = 4
LR = 0.005


def optimize(loss):
    """
    This function calculates the gradients
    :param loss: the loss
    :return: the gradient
    """
    global_step = tf.Variable(0, trainable=False)
    # TBD - consider to use
    learning_rate = tf.train.exponential_decay(LR, global_step, NOF_BATCHES_PER_EPOCH, 0.5, staircase=True)
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grad = opt.minimize(loss)
    return grad


def loss(actions_prob, actions, rewards):
    """
    This function calculates the loss
    :param actions: list of chosen actions
    :param actions_prob: the new measurement from the environment - first layer of the network (array size of 8)
    :param rewards: list of rewards
    :return:
    """

    # Create one hot vector [NOF actions, 4]
    one_hot_vector = tf.one_hot(actions, OUPUT_SIZE)
    # Pick the chosen actions
    chosen_actions = tf.multiply(one_hot_vector, actions_prob)
    # Sum the actions
    chosen_actions_sum = tf.reduce_sum(chosen_actions, axis=1)

    # Compute the left factor loss
    left_factor_loss = tf.log(chosen_actions_sum)

    # Compute the total loss
    loss = - tf.reduce_sum(tf.multiply(left_factor_loss, rewards))
    tf.summary.scalar("loss", loss)
    # Compute gradients
    gradient_step = optimize(loss)

    return loss, gradient_step


def build_model(input):
    """
    This function creates the model (in RL is the agent) which is a simple 3 layer neural network
    :return: the network
    """

    # Weight initializations

    # Define the initialization of the first layer
    w_1 = tf.get_variable(name="w1", shape=(INPUT_SIZE, H1_SIZE),
                          initializer=tf.truncated_normal_initializer())
    b_1 = tf.get_variable(name="b1", shape=(H1_SIZE),
                          initializer=tf.truncated_normal_initializer())

    # Define the initialization of the second layer
    w_2 = tf.get_variable(name="w2", shape=(H1_SIZE, H2_SIZE),
                          initializer=tf.truncated_normal_initializer())
    b_2 = tf.get_variable(name="b2", shape=(H2_SIZE),
                          initializer=tf.truncated_normal_initializer())

    # Define the initialization of the third layer
    w_3 = tf.get_variable(name="w3", shape=(H2_SIZE, OUPUT_SIZE),
                          initializer=tf.truncated_normal_initializer())
    b_3 = tf.get_variable(name="b3", shape=(OUPUT_SIZE),
                          initializer=tf.truncated_normal_initializer())

    # Create neural network
    h1 = tf.nn.tanh(tf.matmul(input, w_1) + b_1, name="h1")
    h2 = tf.nn.tanh(tf.matmul(h1, w_2) + b_2, name="h2")
    y = tf.nn.softmax(tf.matmul(h2, w_3) + b_3, name="y")

    # Output - Softmax
    return y
