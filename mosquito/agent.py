import math
import random

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from mosquito.models import Actor, Critic

'''
TODO:
-add entropy maximization
-add network load and save
-add graphing
-add lstm (might want that to be alternate version)
'''

class PPOAgent(nn.Module):
    def __init__(self, config):
        super(PPOAgent, self).__init__()
        self.actor_mu  = Actor(config)
        self.critic = Critic(config)
        self.exploration = nn.Parameter(torch.zeros(1, config.model_output_shape))

    def get_action(self, x, action=None):
        mu = self.actor_mu(x)
        std = self.exploration.expand_as(mu).exp()
        policy_dist = Normal(mu, std)
        if action is None:
            action = policy_dist.sample()
        log_probs = policy_dist.log_prob(action)
        entropy = policy_dist.entropy()
        return action, log_probs, entropy

    def get_value(self, x):
        return self.critic(x)
