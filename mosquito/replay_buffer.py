import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, config):
        self.i = 0
        self.batch_size = config.batch_size
        self.rollout_length = config.rollout_length

        self.states     = torch.zeros((self.batch_size, self.rollout_length+1, config.model_input_shape  ),dtype=torch.float32)
        self.actions    = torch.zeros((self.batch_size, self.rollout_length+1, config.model_output_shape ),dtype=torch.float32)
        self.log_probs  = torch.zeros((self.batch_size, self.rollout_length+1, config.model_output_shape ),dtype=torch.float32)
        self.values     = torch.zeros((self.batch_size, self.rollout_length+1, 1                         ),dtype=torch.float32)
        self.rewards    = torch.zeros((self.batch_size, self.rollout_length+1, 1                         ),dtype=torch.float32)
        self.done_masks = torch.zeros((self.batch_size, self.rollout_length+1, 1                         ),dtype=torch.float32)
        self.entropies  = torch.zeros((self.batch_size, self.rollout_length+1, 1                         ),dtype=torch.float32)
        self.advantages = torch.zeros((self.batch_size, self.rollout_length+1, 1                         ),dtype=torch.float32)
        self.returns    = torch.zeros((self.batch_size, self.rollout_length+1, 1                         ),dtype=torch.float32)

    def add(self, state, action, log_prob, entropy, value, reward, done):
        ''' This is where you add one transition at a time.'''
        x = self.i % self.rollout_length + 1
        y = self.i // (self.rollout_length + 1)

        self.states[y, x]     = torch.tensor(state,     dtype=torch.float32)
        self.actions[y, x]    = torch.tensor(action,    dtype=torch.float32)
        self.log_probs[y, x]  = torch.tensor(log_prob,  dtype=torch.float32)
        self.values[y, x]     = value
        self.rewards[y, x]    = reward
        self.done_masks[y, x] = 1 - done
        self.entropies[y, x]  = entropy

        self.i += 1

    def batch_add(self, chunk_size, state, action, log_prob, entropy, value, reward, done):
        ''' This is where you add a chunk of transitions at a time.
                For use with vector envs, or multiprocessing'''
        raise NotImplementedError

    def compute_gae(self):
        gae = torch.zeros((self.batch_size, 1)).to(self.agent.device)
        for i in reversed(range(self.rollout_length)):
            delta = self.rewards[:, i] + self.gamma * self.values[:, i+1] * self.done_masks[:, i] - self.values[:, i]
            gae = delta + self.gamma * self.tau * self.done_masks[:, i] * gae
            self.returns[:, i]    = gae + self.values[:, i]
            self.advantages[:, i] = gae

    def random_batch_iter(self, device):
        self.states     = self.states.to(device)    
        self.actions    = self.actions.to(device)   
        self.log_probs  = self.log_probs.to(device) 
        self.values     = self.values.to(device)    
        self.rewards    = self.rewards.to(device)   
        self.done_masks = self.done_masks.to(device)
        self.entropies  = self.entropies.to(device) 
        self.advantages = self.advantages.to(device)
        self.returns    = self.returns.to(device)   

        batch_indices = torch.randperm(self.rollout_length)
        for i in range(self.rollout_length):
            index     = batch_indices[i]
            state     = self.states[:, index]
            action    = self.actions[:, index]
            log_prob  = self.log_probs[:, index]
            entropy   = self.entropies[:, index]
            advantage = self.advantages[:, index]
            return_   = self.returns[:, index]
            yield state, action, log_prob, entropy, advantage, return_