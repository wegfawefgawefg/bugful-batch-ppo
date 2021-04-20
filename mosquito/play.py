import numpy as np
import torch

'''TODO:
        -figure out if replay buffer should be on gpu by default, 
            -should info be .cpu() before going in at all.
'''

class PlayStats:
    def __init__(self) -> None:
        self.ep_lengths = []
        self.scores = []

def play(config, agent, env, n_steps=None, n_eps=None, buffer=None):
    """Play n_samples or n_episodes worth of samples.
    :param n_steps: playing will stop after n steps
    :param n_eps: playing will stop after n episodes
    :param ReplayBuffer buffer: transitions may be saved into this
    .. note::
        kill yourself
    """
    if not n_eps and not n_steps:
        n_eps = 1

    play_stats = PlayStats()

    eps = 0
    steps_this_episode = 0
    total_steps = 0
    score = 0

    state = env.reset()
    with torch.no_grad():
        while True:
            env.render()
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(config.device)
            action, log_prob, entropy = agent.get_action(x)
            value = agent.get_value(x)

            cpu_action = action.cpu().numpy()[0]
            if config.action_clamping == "clip":
                cpu_action = np.clip(cpu_action, -1.0, 1.0)
            elif config.action_clamping == "tanh":
                cpu_action = np.tanh(cpu_action)
            if config.action_scaling:
                assert np.min(cpu_action) >= -1.0 and np.max(cpu_action) <= 1.0, \
                    "action scaling only accepts raw action range = [-1, 1]"
                low, high = config.action_space.low, config.action_space.high
                cpu_action = low + (high - low) * (cpu_action + 1.0) / 2.0  # type: ignore

            state_, reward, done, info = env.step(cpu_action)
            if buffer:
                buffer.add(state, cpu_action, 
                    log_prob.cpu().numpy()[0], 
                    entropy.cpu().item(), 
                    value.cpu().item(), 
                    reward, done)
            
            state = state_
            steps_this_episode += 1
            total_steps += 1
            score += reward

            if done: 
                state = env.reset()
                eps += 1
                
                #   attend to play_stats
                play_stats.scores.append(score)
                score = 0
                play_stats.ep_lengths.append(steps_this_episode)
                steps_this_episode = 0

            if total_steps == n_steps: break
            if eps == n_eps: break

    return buffer, play_stats