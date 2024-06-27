from gym.envs.registration import register

register(
    id='Lattice2D-3actionStateEnv-v0',
    entry_point='gym_lattice.envs:ThreeActionStateEnv',
)
