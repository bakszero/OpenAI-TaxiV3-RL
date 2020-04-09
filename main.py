from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v3')
env = gym.wrappers.Monitor(env, './video/',video_callable=lambda episode_id: True,force = True)

agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)