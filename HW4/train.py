#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import gym
import numpy as np
import cPickle
import tensorflow as tf

from model import build_model, loss


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1, dtype=tf.float64)
    return tf.Variable(weights)


if __name__ == '__main__':

    env_d = 'LunarLander-v2'
    env = gym.make(env_d)
    env.reset()

    # Define PlaceHolders for the graphs
    observation_ph = tf.placeholder(dtype=tf.float32, shape=(None, 8), name="observation_ph")
    action_ph = tf.placeholder(dtype=tf.int32, shape=(None), name="actions_ph")
    reward_ph = tf.placeholder(dtype=tf.float32, shape=(None), name="rewards_ph")

    # Our agent is a neural network
    agent = build_model(observation_ph)

    # Create the loss to the graph
    loss, gradient_step = loss(agent, action_ph, reward_ph)

    episode_number = 0
    total_episodes = 100000
    # Initialize the Computational Graph
    init = tf.global_variables_initializer()

    # Lists
    action_lst = []
    observation_lst = []
    reward_lst = []

    # Define Summaries
    summary_writer = tf.summary.FileWriter('logs/')
    summaries = tf.summary.merge_all()

    with tf.Session() as sess:

        sess.run(init)
        obsrv = env.reset()  # Obtain an initial observation of the environment

        while episode_number <= total_episodes:
            # Run the policy network and get a distribution over actions
            action_probs = sess.run(agent, feed_dict={observation_ph: np.expand_dims(obsrv, axis=0)})

            # Sample action from distribution
            action = np.argmax(np.random.multinomial(1, action_probs.flatten()))

            # Load the gui
            env.render()

            # Step the environment and get new measurements
            obsrv, reward, done, info = env.step(action)

            reward_lst.append(reward)
            reward_lst = [np.sum(reward_lst[i:]) for i in range(len(reward_lst))]
            action_lst.append(action)
            observation_lst.append(obsrv)

            if done:
                episode_number += 1
                loss_val, _, summary_val = sess.run([loss, gradient_step.summaries],
                                                    feed_dict={action_ph: action_lst, reward_ph: reward_lst,
                                                               observation_ph: observation_lst})
                # Add summary
                summary_writer.add_summary(summary_val, global_step=episode_number)
                print("Episode: {0}, Loss: {1}".format(episode_number, loss_val))
                obsrv = env.reset()
                action_lst = []
                reward_lst = []
                observation_lst = []

    print("End training")
