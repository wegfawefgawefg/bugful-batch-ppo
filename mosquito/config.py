import os
import datetime

import gym
import torch
import numpy as np

import gym_continuous_cartpole

class Config:
    def __init__(self):
        """The config for training.
        :setting action_clamping: chops/scales the agent actions to be [-1,1] for the env
            options: "clip" or "tanh"
        :setting action_scaling: 
            for continuous action spaces, if action space outside [-1, 1] set to True
            for discrete action spaces or continuous action spaces in range [-1, 1] set to False
        """
        #   general settings
        self.run_title = "mosquito-v0.1"
        self.layers = [256, 256]
        self.learn_rate = 1e-3
        self.rollout_length = 8
        self.batch_size = 32
        self.n_samples_collect_per_train = self.batch_size * (self.rollout_length + 1)
        self.num_test_eps = 1
        self.gpu = True

        #   env settings
        self.env = "CartPoleContinuous-v0"  #"Pendulum-v0"
        self.action_clamping = "clip"
        self.action_scaling = False

        #   ppo settings
        self.policy_iter_epochs = 8
        self.entropy_weight     = 0.001
        self.kl_clip            = 0.2
        self.gamma              = 0.7
        self.tau                = 0.7

        #   checkpoints
        self.make_checkpoints = False
        self.checkpoints_dir = "/media/vega/ThinThicc/_SSD_CODEING/python/plan/checkpoints"        

    def validate(self):
        def alert_invalid_config():
            print("#"*40)
            print("Invalid Config:")
            print("-"*40)

        n_errors = 0
        
        if self.gpu == True:
            if not torch.cuda.is_available():
                alert_invalid_config()
                print("Device does not support gpu training...")
                n_errors += 1
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        try:
            env = gym.make(self.env)
        except Exception as e:
            alert_invalid_config()
            print("Cant make env...")
            print(e)
            n_errors += 1

        try:
            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.model_input_shape = int(np.array(self.observation_space.shape).prod())
            self.model_output_shape = int(np.array(self.action_space.shape).prod())
            #print(f"observation_space: {self.observation_space}")
            #print(f"action_space: {self.action_space}")
            #print(f"model_input_shape: {self.model_input_shape}")
            #print(f"model_output_shape: {self.model_output_shape}")
        except Exception as e:
            alert_invalid_config()
            print("Obs/Action space error...")
            print(e)
            n_errors += 1

        try:
            date_section = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            self.run_title = f"{self.run_title}_{self.env}_{date_section}"
            self.this_checkpoint_dir = os.path.join(self.checkpoints_dir, self.run_title)
            if not os.path.exists(self.checkpoints_dir):
                os.makedirs(self.checkpoints_dir)
            os.makedirs(self.this_checkpoint_dir)
            os.makedirs(os.path.join(self.this_checkpoint_dir, "peak_score"))
            os.makedirs(os.path.join(self.this_checkpoint_dir, "periodic"))
        except Exception as e:
            alert_invalid_config()
            print("Cant setup checkpoints directory...")
            print(e)
            n_errors += 1

        if not n_errors == 0:
            alert_invalid_config()
            print(f"{n_errors} config errors!")
