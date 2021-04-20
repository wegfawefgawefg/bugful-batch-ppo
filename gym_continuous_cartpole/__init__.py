from gym.envs.registration import register

register(
    id='CartPoleContinuous-v0',
    entry_point='gym_continuous_cartpole.env.continuous_cartpole:ContinuousCartPoleEnv',
    reward_threshold=0.15,
)
