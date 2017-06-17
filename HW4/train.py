#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import gym
import numpy as np
import cPickle
import tensorflow as tf


__author__ = 'roeih'


def agent(observation):
    """
    This function creates the agent which is a simple 3 layer neural network
    :return: the network
    """

    # Define layers size
    input_size = 8
    h1_size = 15
    h2_size = 15
    output_size = 4

    # Weight initializations
    w_1 = init_weights((input_size, h1_size))
    w_2 = init_weights((h1_size, h2_size))
    w_3 = init_weights((h2_size, output_size))

    # Create neural network
    h1 = tf.nn.sigmoid(tf.matmul(tf.reshape(observation, shape=(1, 8)), w_1), name="h1")
    h2 = tf.nn.sigmoid(tf.matmul(h1, w_2), name="h2")
    h3 = tf.nn.sigmoid(tf.matmul(h2, w_3), name="h3")
    # Output - Softmax
    y = tf.nn.softmax(h3, name="y")
    return y


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1, dtype=tf.float64)
    return tf.Variable(weights)


if __name__ == '__main__':

    env_d = 'LunarLander-v2'
    env = gym.make(env_d)
    env.reset()

    observation = tf.placeholder(dtype=tf.float64, shape=(8), name="observation")
    y = agent(observation)

    init = tf.global_variables_initializer()

    episode_number = 0
    total_episodes = 100

    with tf.Session() as sess:
        sess.run(init)
        obsrv = env.reset()  # Obtain an initial observation of the environment
        while episode_number <= total_episodes:
            # Run the policy network and get a distribution over actions
            action_probs = sess.run(y, feed_dict={observation: obsrv})

            # Sample action from distribution
            action = np.argmax(np.random.multinomial(1, action_probs.flatten()))

            # Step the environment and get new measurements
            obsrv, reward, done, info = env.step(action)

            print ("observ: {}", obsrv)
            print ("reward: {}", reward)

            if done:
                print("Done")
                episode_number += 1
                obsrv = env.reset()

    tf.app.run()
