#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 23:11:02 2017

@author:
"""
import gym
import numpy as np
import cPickle
import tensorflow as tf

env_d = 'LunarLander-v2'
env = gym.make(env_d)
env.reset()

observation = tf.placeholder()


def agent(observation):
    return 0


y = agent(observation)

init = tf.global_variables_initializer()


def main(argv):
    with tf.Session() as sess:
        sess.run(init)
        obsrv = env.reset()  # Obtain an initial observation of the environment
        while episode_number <= total_episodes:
            # Run the policy network and get a distribution over actions
            action_probs = sess.run(y, feed_dict={observation: obsrv})
            # sample action from distribution
            action = np.argmax(np.multinomial(1, action_probs))
            # step the environment and get new measurements
            obsrv, reward, done, info = env.step(action)
            if done:
                episode_number += 1
                obsrv = env.reset()


if __name__ == '__main__':
    tf.app.run()
