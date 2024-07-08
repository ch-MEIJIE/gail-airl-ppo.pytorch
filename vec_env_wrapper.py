import PyFlyt.gym_envs  # noqa
import gymnasium as gym
import numpy as np


class VecPyFlytEnvWrapper:
    def __init__(
        self,
        render_mode: str = "human",
        env_id: str = "PyFlyt/QuadX-UVRZ-Gates-v2",
        num_env: int = 5,
        seed: int = 0
    ) -> None:
        self.env = gym.make_vec(
            env_id,
            render_mode=render_mode,
            agent_hz=2,
            seed=seed,
            num_envs=num_env,
            vectorization_mode="async",
            vector_kwargs=({'shared_memory': False})
        )
        self.num_env = num_env
        self.targets_num = self.env.get_attr('targets_num')[0]
        self.act_size = self.env.get_attr('action_space')[0].n
        obs_space = self.env.get_attr('observation_space')[0]
        self.obs_atti_size = obs_space['attitude'].shape[0]
        self.obs_target_size = obs_space['target_deltas'].feature_space.shape[0]

        # TODO: Flatten the target delta bound space in ENV
        self.obs_bound_size = obs_space["target_delta_bound"].shape[0]
        self._max_episode_steps = self.env.get_attr('max_steps')[0]

    def reset(self):
        obs, _ = self.env.reset()
        self.state_atti = obs['attitude']
        self.state_targ = np.zeros(
            (self.num_env, self.obs_target_size)
        )
        for i in range(self.num_env):
            self.state_targ[i] = obs['target_deltas'][i][0]
        self.state_bound = obs['target_delta_bound']

        obs = self.concat_state()

        return obs

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)

        self.state_atti = obs['attitude']
        self.state_targ = np.zeros(
            (self.num_env, self.obs_target_size)
        )
        for i in range(self.num_env):
            self.state_targ[i] = obs['target_deltas'][i][0]
        self.state_bound = obs['target_delta_bound']

        obs = self.concat_state()
        done = np.logical_or(term, trunc)

        return obs, reward, done, info

    def concat_state(self):
        return np.concatenate(
            [self.state_atti, self.state_targ, self.state_bound], axis=1
        )
