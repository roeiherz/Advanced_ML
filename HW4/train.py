#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import gym
import numpy as np
import cPickle
import tensorflow as tf
import os

from model import build_model, loss, OUPUT_SIZE

# discount factor of theb rewards
NOF_BATCHES_PER_EPOCH = 20
SAVE_MODEL_NAME = "model_lr_decay.ckpt"
LOAD_MODEL_NAME = "model_lr_decay.ckpt"
GAMMA = 0.99
# number of episodes per atch
BATCH_SIZE = 30
# use saved model
USE_SAVED_MODEL = True
# number of steps in each episode (will be increased)
SIZE = 1000
# total episodes
TOTAL_EPISODES = 30000


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

    episode_number = 1
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

    # check the commutative rewards
    cum_rwrd = 0.

    with tf.Session() as sess:

        # Restore variables from disk.
        if os.path.exists(LOAD_MODEL_NAME + ".index") and USE_SAVED_MODEL:
            saver.restore(sess, "./" + LOAD_MODEL_NAME)
            print("Model restored.")
        else:
            sess.run(init)

        # Init gym environment
        env.mode = 'fast'
        obsrv = env.reset()  # Obtain an initial observation of the environment

        while episode_number <= TOTAL_EPISODES:

            # holds the normalized rewards of the all batch
            rewards_mat = np.zeros((0))

            while episode_number % BATCH_SIZE != 0:
                # Run the policy network and get a distribution over actions
                action_probs = sess.run(agent, feed_dict={observation_ph: np.expand_dims(obsrv, axis=0)})

                # Sample action from distribution
                # action = np.argmax(np.random.multinomial(1, action_probs.flatten()))
                action = np.random.choice(np.arange(0, OUPUT_SIZE), p=np.squeeze(action_probs))
                # Load the gui
                env.render()

                # Step the environment and get new measurements
                observation_lst.append(obsrv)

                obsrv, reward, done, info = env.step(action)
                cum_rwrd += reward
                reward_lst.append(reward)
                action_lst.append(action)

                # episode done
                if done or len(reward_lst) == SIZE:
                    episode_number += 1

                    # discount and normalize
                    rewards_episode_mat = np.asarray(reward_lst)
                    episode_avg_reward = np.sum(rewards_episode_mat)
                    avg = cum_rwrd / (episode_number - 1)
                    print "Episode {0} Sum Rewards: {1} Avg Rewards: {2}".format(episode_number, episode_avg_reward,
                                                                                 avg)

                    # Gamma Factor
                    for i in range(len(reward_lst)):
                        rewards_episode_mat[i] = np.sum(
                            rewards_episode_mat[i:] * np.power(GAMMA, np.arange(0, len(reward_lst) - i)))

                    # Normalize Gamma
                    rewards_episode_std = np.std(rewards_episode_mat)
                    rewards_episode_mat = (rewards_episode_mat - np.mean(rewards_episode_mat)) / rewards_episode_std

                    # append to matrix of the all batch
                    rewards_mat = np.concatenate((rewards_mat, rewards_episode_mat))

                    # init per episode reward list
                    reward_lst = []

                    # start new episode
                    obsrv = env.reset()

            # minimize loss according to batch
            loss_val, _, summary_val = sess.run([loss, gradient_step, summaries],
                                                feed_dict={action_ph: action_lst, reward_ph: rewards_mat,
                                                           observation_ph: observation_lst})
            # Add summary
            summary_writer.add_summary(summary_val, global_step=episode_number)
            print("Episode: {0}, Loss: {1}".format(episode_number, loss_val))

            # increase number of allowed steps per episode
            if episode_number % (NOF_BATCHES_PER_EPOCH * BATCH_SIZE) == 0:
                SIZE *= 2

            # save the model
            if episode_number % (NOF_BATCHES_PER_EPOCH * BATCH_SIZE) == 0:
                # Save the variables to disk.
                save_path = saver.save(sess, SAVE_MODEL_NAME)
                print("Model saved in file: %s" % save_path)

                # dump weights to pickle file (used by the tester)
                tvars = tf.trainable_variables()
                param = sess.run(tvars)
                filename = 'ws.p'
                weights_file = open(filename, 'wb')
                cPickle.dump(param, weights_file)
                weights_file.close()

            # start new batch
            obsrv = env.reset()
            action_lst = []
            reward_lst = []
            observation_lst = []

            episode_number += 1

    print("End training")
