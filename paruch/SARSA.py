from collections import deque
import gym
import random
import numpy as np
import time
import pickle

from collections import defaultdict


EPISODES =   20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999



def default_Q_value():
    return 0


if __name__ == "__main__":




    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v0")
    env.seed(1)
    env.action_space.np_random.seed(1)


    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.

    episode_reward_record = deque(maxlen=100)


    for i in range(EPISODES):
        episode_reward = 0
       
        #TODO perform SARSA learning

        obs = env.reset() # save original environment
        done = False # runs the while loop below

        while done == False:
            if random.uniform(0,1) < EPSILON:
                action = env.action_space.sample()
            else:
                prediction = np.array([Q_table[(obs, i)] for i in range(env.action_space.n)])
                action =  np.argmax(prediction)

            state = obs # saves original environment

            obs,reward,done,info = env.step(action) # steps into an action
            episode_reward += reward # adds reward to episode_reward

            optimal = max([Q_table[(obs,i)] for i in range(env.action_space.n)]) # finds max Q(s', a')

            if not done: # if not done ...
              # updating Q_table
              Q_table[state, action] += LEARNING_RATE * \
              (reward + (DISCOUNT_FACTOR * optimal) - Q_table[state, action])

            if done: # if done (once terminal state is found) ..
              # update Q_table
              Q_table[state, action] += LEARNING_RATE * (reward - Q_table[state, action])
              episode_reward_record.append(episode_reward) # append total episode reward to episode_reward_record

        EPSILON *= EPSILON_DECAY # decay epsilon

        if i%100 ==0 and i>0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    ####DO NOT MODIFY######
    model_file = open('SARSA_Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    #######################



