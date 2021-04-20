import torch
import numpy as np

import gym

from mosquito import (Config, Stats, PPOAgent, ReplayBuffer, 
    play, train_ppo,
)

config = Config()
config.validate()
stats = Stats()
agent = PPOAgent(config).to(config.device)
optimizer = torch.optim.Adam(agent.parameters(), lr=config.learn_rate)
env = gym.make(config.env)
while True:
    '''     collect     '''
    buffer, play_stats = play(config, agent, env, 
        n_steps=config.n_samples_collect_per_train, buffer=ReplayBuffer(config))
    stats.update_collection_stats(num_new_samples_collected=sum(play_stats.ep_lengths))
    '''     train       '''
    train_stats = train_ppo(config, agent, optimizer, buffer)
    stats.update_training_stats(num_new_samples_processed=\
        config.policy_iter_epochs*stats.num_samples_collected)
    '''     test        '''
    _, play_stats = play(config, agent, env, n_eps=config.num_test_eps)
    stats.update_test_stats(
        num_new_test_eps=config.num_test_eps, 
        latest_test_score=play_stats.scores[-1])
    stats.print_test_run_stats()
