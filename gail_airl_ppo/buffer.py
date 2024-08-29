import os
import numpy as np
import torch


class SerializedBuffer:

    def __init__(self, path, device):
        tmp = torch.load(path)
        self.buffer_size = self._n = tmp['state'].size(0)
        self.device = device

        self.states = tmp['state'].clone().to(self.device)
        self.actions = tmp['action'].clone().to(self.device)
        self.rewards = tmp['reward'].clone().to(self.device)
        self.dones = tmp['done'].clone().to(self.device)
        self.next_states = tmp['next_state'].clone().to(self.device)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes]
        )


class Buffer(SerializedBuffer):

    def __init__(self, buffer_size, state_shape, action_shape, device):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.device = device

        self.states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        torch.save({
            'state': self.states.clone().cpu(),
            'action': self.actions.clone().cpu(),
            'reward': self.rewards.clone().cpu(),
            'done': self.dones.clone().cpu(),
            'next_state': self.next_states.clone().cpu(),
        }, path)


class RolloutBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device, mix=1):
        self._n = 0
        self._p = 0
        self.mix = mix
        self.buffer_size = buffer_size
        self.total_size = mix * buffer_size

        self.states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (self.total_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, log_pi, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def get(self):
        assert self._p % self.buffer_size == 0
        start = (self._p - self.buffer_size) % self.total_size
        idxes = slice(start, start + self.buffer_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )

    def sample(self, batch_size):
        assert self._p % self.buffer_size == 0
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )


class VecRolloutBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device, num_env, mix=1):
        self._n = 0
        self._p = 0
        self.mix = mix
        self.buffer_size = buffer_size
        self.total_size = mix * buffer_size

        self.states = torch.empty(
            (self.total_size, num_env, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (self.total_size, num_env), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (self.total_size, num_env), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (self.total_size, num_env), dtype=torch.float, device=device)
        self.log_pis = torch.empty(
            (self.total_size, num_env), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (self.total_size, num_env, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, log_pi, next_state):
        # considering the environment dimension, append the data
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p].copy_(torch.from_numpy(reward))
        self.dones[self._p].copy_(torch.Tensor(done))
        self.log_pis[self._p].copy_(torch.from_numpy(log_pi))
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def get(self):
        assert self._p % self.buffer_size == 0
        start = (self._p - self.buffer_size) % self.total_size
        idxes = slice(start, start + self.buffer_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )

    def sample(self, batch_size):
        assert self._p % self.buffer_size == 0
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )


class SeqReplayBuffer:
    buffer_type = "seq_vanilla"

    def __init__(
        self,
        max_replay_buffer_size,
        observation_dim,
        action_dim,
        sampled_seq_len: int,
        sample_weight_baseline: float,
        device: str = "cpu",
        **kwargs
    ):
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self.device = torch.device(device)

        self._observations = torch.zeros(
            (max_replay_buffer_size,
             observation_dim), dtype=torch.float, device=device
        )
        self._next_observations = torch.zeros(
            (max_replay_buffer_size,
             observation_dim), dtype=torch.float, device=device
        )
        self._actions = torch.zeros(
            (max_replay_buffer_size, action_dim), dtype=torch.float, device=device
        )
        self._rewards = torch.zeros(
            (max_replay_buffer_size, 1), dtype=torch.float, device=device
        )
        self._terminals = torch.zeros(
            (max_replay_buffer_size, 1), dtype=torch.int64, device=device
        )
        self._valid_starts = torch.zeros(
            (max_replay_buffer_size), dtype=torch.float, device=device
        )

        assert sampled_seq_len >= 2
        assert sample_weight_baseline >= 0.0
        self._sampled_seq_len = sampled_seq_len
        self._sample_weight_baseline = sample_weight_baseline

        self.clear()

        RAM = 0.0
        for name, var in vars(self).items():
            if isinstance(var, torch.Tensor):
                RAM += var.element_size() * var.nelement()
        print(f"buffer RAM usage: {RAM / 1024 ** 3 :.2f} GB")

    def size(self):
        return self._size

    def clear(self):
        self._top = 0
        self._size = 0

    def add_episode(self, observations, actions, rewards, terminals, next_observations):
        assert (
            observations.shape[0]
            == actions.shape[0]
            == rewards.shape[0]
            == terminals.shape[0]
            == next_observations.shape[0]
            >= 2
        )

        seq_len = observations.shape[0]
        indices = torch.arange(self._top, self._top +
                               seq_len) % self._max_replay_buffer_size

        self._observations[indices] = observations
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)
        self._actions[indices] = actions
        self._rewards[indices] = rewards
        self._terminals[indices] = terminals
        self._next_observations[indices] = next_observations

        self._valid_starts[indices] = self._compute_valid_starts(seq_len)

        self._top = (self._top + seq_len) % self._max_replay_buffer_size
        self._size = min(self._size + seq_len, self._max_replay_buffer_size)

    def _compute_valid_starts(self, seq_len):
        valid_starts = torch.ones(
            seq_len, dtype=torch.float, device=self.device)

        num_valid_starts = float(
            max(1.0, seq_len - self._sampled_seq_len + 1.0))
        total_weights = self._sample_weight_baseline + num_valid_starts
        valid_starts *= total_weights / num_valid_starts
        valid_starts[int(num_valid_starts):] = 0.0

        return valid_starts

    def random_episodes(self, batch_size):
        sampled_episode_starts = self._sample_indices(batch_size)

        indices = []
        for start in sampled_episode_starts:
            end = start + self._sampled_seq_len
            indices += list(torch.arange(start, end) %
                            self._max_replay_buffer_size)

        batch = self._sample_data(indices)
        masks = self._generate_masks(indices, batch_size)
        batch["mask"] = masks

        for k in batch.keys():
            batch[k] = batch[k].reshape(
                batch_size, self._sampled_seq_len, -1).transpose(0, 1)

        return batch

    def _sample_indices(self, batch_size):
        valid_starts_indices = torch.where(self._valid_starts > 0.0)[0]
        sample_weights = self._valid_starts[valid_starts_indices].clone()
        sample_weights /= sample_weights.sum()

        return torch.multinomial(sample_weights, batch_size, replacement=True)

    def _sample_data(self, indices):
        return dict(
            obs=self._observations[indices],
            act=self._actions[indices],
            rew=self._rewards[indices],
            term=self._terminals[indices],
            obs2=self._next_observations[indices],
        )

    def _generate_masks(self, indices, batch_size):
        sampled_seq_valids = self._valid_starts[indices].clone().reshape(
            batch_size, self._sampled_seq_len
        )
        sampled_seq_valids[sampled_seq_valids > 0.0] = 1.0

        masks = torch.ones_like(sampled_seq_valids)

        diff = sampled_seq_valids[:, :-1] - sampled_seq_valids[:, 1:]
        diff = torch.cat(
            [torch.ones(batch_size, 1, device=self.device), diff], dim=1)

        indices_array = torch.tensor(indices).reshape(
            batch_size, self._sampled_seq_len)
        diff[indices_array == self._top] = -1.0

        invalid_starts_b, invalid_starts_t = torch.where(diff == -1.0)
        invalid_indices_b = []
        invalid_indices_t = []
        last_batch_index = -1

        for batch_index, start_index in zip(invalid_starts_b, invalid_starts_t):
            if batch_index == last_batch_index:
                continue
            last_batch_index = batch_index

            invalid_indices = list(range(start_index, self._sampled_seq_len))
            invalid_indices_b += [batch_index] * len(invalid_indices)
            invalid_indices_t += invalid_indices

        masks[invalid_indices_b, invalid_indices_t] = 0.0

        return masks


class SeqReplayBufferParallel:
    buffer_type = "seq_vanilla_parallel"

    def __init__(
        self,
        num_envs,
        max_replay_buffer_size,
        observation_dim,
        action_dim,
        sampled_seq_len: int,
        sample_weight_baseline: float,
        device: str = "cpu",
        **kwargs
    ):
        self.num_envs = num_envs
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self.device = torch.device(device)

        # Adjust tensors to accommodate multiple environments
        self._observations = torch.zeros(
            (num_envs, max_replay_buffer_size,
             observation_dim), dtype=torch.float, device=self.device
        )
        self._next_observations = torch.zeros(
            (num_envs, max_replay_buffer_size,
             observation_dim), dtype=torch.float, device=self.device
        )
        self._actions = torch.zeros(
            (num_envs, max_replay_buffer_size,
             action_dim), dtype=torch.float, device=self.device
        )
        self._rewards = torch.zeros(
            (num_envs, max_replay_buffer_size, 1), dtype=torch.float, device=self.device
        )
        self._terminals = torch.zeros(
            (num_envs, max_replay_buffer_size, 1), dtype=torch.uint8, device=self.device
        )
        self._valid_starts = torch.zeros(
            (num_envs, max_replay_buffer_size), dtype=torch.float, device=self.device
        )

        assert sampled_seq_len >= 2
        assert sample_weight_baseline >= 0.0
        self._sampled_seq_len = sampled_seq_len
        self._sample_weight_baseline = sample_weight_baseline

        self.clear()

        RAM = 0.0
        for name, var in vars(self).items():
            if isinstance(var, torch.Tensor):
                RAM += var.element_size() * var.nelement()
        print(f"buffer RAM usage: {RAM / 1024 ** 3 :.2f} GB")

    def size(self):
        return self._size

    def clear(self):
        self._top = torch.zeros(
            self.num_envs, dtype=torch.int, device=self.device)
        self._size = torch.zeros(
            self.num_envs, dtype=torch.int, device=self.device)

    def add_episode(self, env_id, observations, actions, rewards, terminals, next_observations):
        assert (
            observations.shape[0]
            == actions.shape[0]
            == rewards.shape[0]
            == terminals.shape[0]
            == next_observations.shape[0]
            >= 2
        )

        seq_len = observations.shape[0]
        top = self._top[env_id].item()
        indices = torch.arange(
            top, top + seq_len) % self._max_replay_buffer_size

        self._observations[env_id, indices] = observations
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)
        self._actions[env_id, indices] = actions
        self._rewards[env_id, indices] = rewards
        self._terminals[env_id, indices] = terminals
        self._next_observations[env_id, indices] = next_observations

        self._valid_starts[env_id,
                           indices] = self._compute_valid_starts(seq_len)

        self._top[env_id] = (top + seq_len) % self._max_replay_buffer_size
        self._size[env_id] = min(
            self._size[env_id] + seq_len, self._max_replay_buffer_size)

    def _compute_valid_starts(self, seq_len):
        valid_starts = torch.ones(
            seq_len, dtype=torch.float, device=self.device
        )

        num_valid_starts = float(
            max(1.0, seq_len - self._sampled_seq_len + 1.0))
        total_weights = self._sample_weight_baseline + num_valid_starts
        valid_starts *= total_weights / num_valid_starts
        valid_starts[int(num_valid_starts):] = 0.0

        return valid_starts

    def random_episodes(self, env_id, batch_size):
        sampled_episode_starts = self._sample_indices(env_id, batch_size)

        indices = []
        for start in sampled_episode_starts:
            end = start + self._sampled_seq_len
            indices += list(torch.arange(start, end) %
                            self._max_replay_buffer_size)

        batch = self._sample_data(env_id, indices)
        masks = self._generate_masks(env_id, indices, batch_size)
        batch["mask"] = masks

        for k in batch.keys():
            batch[k] = batch[k].reshape(
                batch_size, self._sampled_seq_len, -1).transpose(0, 1)

        return batch

    def _sample_indices(self, env_id, batch_size):
        valid_starts_indices = torch.where(self._valid_starts[env_id] > 0.0)[0]
        sample_weights = self._valid_starts[env_id][valid_starts_indices].clone(
        )
        sample_weights /= sample_weights.sum()

        return torch.multinomial(sample_weights, batch_size, replacement=True)

    def _sample_data(self, env_id, indices):
        return dict(
            obs=self._observations[env_id, indices],
            act=self._actions[env_id, indices],
            rew=self._rewards[env_id, indices],
            term=self._terminals[env_id, indices],
            obs2=self._next_observations[env_id, indices],
        )

    def _generate_masks(self, env_id, indices, batch_size):
        sampled_seq_valids = self._valid_starts[env_id, indices].clone().reshape(
            batch_size, self._sampled_seq_len
        )
        sampled_seq_valids[sampled_seq_valids > 0.0] = 1.0

        masks = torch.ones_like(sampled_seq_valids)

        diff = sampled_seq_valids[:, :-1] - sampled_seq_valids[:, 1:]
        diff = torch.cat(
            [torch.ones(batch_size, 1, device=self.device), diff], dim=1
        )

        indices_array = torch.tensor(indices).reshape(
            batch_size, self._sampled_seq_len)
        diff[indices_array == self._top[env_id]] = -1.0

        invalid_starts_b, invalid_starts_t = torch.where(diff == -1.0)
        invalid_indices_b = []
        invalid_indices_t = []
        last_batch_index = -1

        for batch_index, start_index in zip(invalid_starts_b, invalid_starts_t):
            if batch_index == last_batch_index:
                continue
            last_batch_index = batch_index

            invalid_indices = list(range(start_index, self._sampled_seq_len))
            invalid_indices_b += [batch_index] * len(invalid_indices)
            invalid_indices_t += invalid_indices

        masks[invalid_indices_b, invalid_indices_t] = 0.0

        return masks
