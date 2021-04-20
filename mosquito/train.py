import torch

def train_ppo(config, agent, optimizer, buffer):
    for _ in range(config.policy_iter_epochs):
        for state, action, old_log_probs, entropy, advantage, return_ in buffer.random_batch_iter(config.device):
            action, new_log_probs, entropy = agent.get_action(state, action=action)
            value = agent.get_value(state)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - config.kl_clip, 1.0 + config.kl_clip) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            entropy = entropy.mean()
            loss = 0.5 * critic_loss + actor_loss - config.entropy_weight * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()