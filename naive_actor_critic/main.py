import gym
import numpy as np
from actor_critic_replay_torch import Agent
from utils import plotLearning

if __name__ == '__main__':
	env = gym.make('LunarLander-v2')
	n_games = 1500
	agent = Agent(gamma=0.99, lr=1e-4, input_dims=[8], n_actions = 4,
				l1_size=256, l2_size=256)

	filename = 'plot.png'
	scores = []

	for i in range(n_games):
		done = False
		observation = env.reset()
		score = 0.0

		while not done:
			action, prob = agent.choose_action(observation)
			observation_, reward, done, info = env.step(action)
			score += reward
			agent.store_transition(observation, prob, reward, 
									observation_, int(done))
			agent.learn()
			observation = observation_

		scores.append(score)
		avg_score = np.mean(scores[-100:])
		print('episode ', i, 'score %.1f' % score, 'avg score %.2f' % avg_score)

	x = [i+1 for i in range(n_games)]
	plotLearning(scores, filename, x)