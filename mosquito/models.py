import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, config):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(config.model_input_shape, config.layers[0]), 
            nn.ReLU(), 
            nn.Linear(config.layers[0],config.layers[1]),  
            nn.ReLU(),
            nn.Linear(config.layers[1],config.model_output_shape),
        )

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, config):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(config.model_input_shape, config.layers[0]), 
            nn.ReLU(), 
            nn.Linear(config.layers[0],config.layers[1]),  
            nn.ReLU(),
            nn.Linear(config.layers[1],1),
        )

    def forward(self, x):
        return self.net(x)