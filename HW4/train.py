#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import gym
import numpy as np
import cPickle
import tensorflow as tf
import os

from model import build_model, loss, OUPUT_SIZE

GAMMA = 0.99

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
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Lists
    action_lst = []
    observation_lst = []
    reward_lst = []

    # Define Summaries
    summary_writer = tf.summary.FileWriter('logs/', graph=tf.get_default_graph())
    summaries = tf.summary.merge_all()

    with tf.Session() as sess:



        # Restore variables from disk.
        if os.path.exists("model.ckpt.index") and False:
            saver.restore(sess, "./model.ckpt")
            print("Model restored.")
        else:
            sess.run(init)

        # Do some work with the model
        env.mode = 'human'
        obsrv = env.reset()  # Obtain an initial observation of the environment
        size = 100
        while episode_number <= total_episodes:
            # Run the policy network and get a distribution over actions
            action_probs = sess.run(agent, feed_dict={observation_ph: np.expand_dims(obsrv, axis=0)})

            # Sample action from distribution
            #action = np.argmax(np.random.multinomial(1, action_probs.flatten()))
            action = np.random.choice(np.arange(0, OUPUT_SIZE), p=np.squeeze(action_probs))
            # Load the gui
            env.render()

            # Step the environment and get new measurements
            obsrv, reward, done, info = env.step(action)

            reward_lst.append(reward * np.power(GAMMA, len(reward_lst)))
            action_lst.append(action)
            observation_lst.append(obsrv)

            if done or len(reward_lst) == size:
                episode_number += 1

                rewards_mat = np.asarray([np.sum(reward_lst[i:]) for i in range(len(reward_lst))])
                rewards_mat = (rewards_mat - np.mean(rewards_mat))/np.std(rewards_mat)
                loss_val, _, summary_val = sess.run([loss, gradient_step, summaries],
                                                    feed_dict={action_ph: action_lst, reward_ph: rewards_mat,
                                                               observation_ph: observation_lst})
                # Add summary
                summary_writer.add_summary(summary_val, global_step=episode_number)
                print("Episode: {0}, Loss: {1}".format(episode_number, loss_val))
                obsrv = env.reset()
                action_lst = []
                reward_lst = []
                observation_lst = []
                obsrv = env.reset()

                if episode_number % 1000 == 0:
                    size *= 2

                # save the model
                if episode_number % 1000 == 0:
                    # Save the variables to disk.
                    save_path = saver.save(sess, "model.ckpt")
                    print("Model saved in file: %s" % save_path)

                    # dump weights to pickle file (used by the tester)
                    tvars = tf.trainable_variables()
                    param = sess.run(tvars)
                    filename = 'ws.p'
                    weights_file = open(filename, 'wb')
                    cPickle.dump(param, weights_file)
                    weights_file.close()

    print("End training")
