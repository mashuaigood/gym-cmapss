from gym.envs.registration import register

register(
    id='cmapss-v0',
    entry_point='gym_cmapss.envs:CMAPSSEnv',
)