import torch
import os
from torch import nn
from torch.optim import Adam

from .base import Algorithm
from gail_airl_ppo.buffer import VecRolloutBuffer
from gail_airl_ppo.network import StateIndependentPolicyDiscrete, StateFunction


def calculate_gae(values, rewards, dones, next_values, gamma, lambd):
    values = values.squeeze(-1)
    next_values = next_values.squeeze(-1)
    gaes = torch.empty_like(rewards, dtype=torch.float32)
    for env_idx in range(rewards.size(1)):
        deltas = rewards[:, env_idx]\
            + gamma * next_values[:, env_idx]\
            * (1 - dones[:, env_idx]) - values[:, env_idx]
        gaes[-1, env_idx] = deltas[-1]
        for t in reversed(range(rewards.size(0) - 1)):
            gaes[t, env_idx] = deltas[t] + gamma * lambd * \
                (1 - dones[t, env_idx]) * gaes[t + 1, env_idx]
    gaes = (gaes - gaes.mean(dim=0, keepdim=True)) / \
        (gaes.std(dim=0, keepdim=True) + 1e-8)
    return gaes+values, gaes


class PPO(Algorithm):

    def __init__(self, state_shape, action_shape, device, seed, num_env,
                 context_length=1, gamma=0.995, rollout_length=100,
                 mix_buffer=20, lr_actor=3e-4,
                 lr_critic=3e-4, units_actor=(64, 64), units_critic=(64, 64),
                 epoch_ppo=10, clip_eps=0.2, lambd=0.97, coef_ent=0.0,
                 max_grad_norm=10.0):
        super().__init__(state_shape, action_shape, device, seed, gamma)

        # Rollout buffer.
        # flatten_state_shape = state_shape[0] + \
        #     state_shape[1][0]*state_shape[1][1] + state_shape[2]
        self.buffer = VecRolloutBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=[1],
            device=device,
            num_env=num_env,
            mix=mix_buffer
        )

        # Actor.
        self.actor = StateIndependentPolicyDiscrete(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.Tanh()
        ).to(device)

        # Critic.
        self.critic = StateFunction(
            state_shape=state_shape,
            hidden_units=units_critic,
            hidden_activation=nn.Tanh()
        ).to(device)

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)

        self.learning_steps_ppo = 0
        self.rollout_length = rollout_length
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm

    def is_update(self, step):
        return step % self.rollout_length == 0

    def explore(self, state):
        '''Stochastic Policy'''
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float).to(self.device)
        with torch.no_grad():
            action, action_logprob = self.actor.act(state)
        return action.cpu().numpy(), action_logprob.cpu().numpy()

    def exploit(self, state):
        '''Deterministic Policy'''
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float).to(self.device)
        with torch.no_grad():
            action = self.actor.exploit(state)
        return action.cpu().numpy()

    def step(self, env, state, t, step):
        t += 1

        action, log_pi = self.explore(state)

        next_state, reward, dones, _ = env.step(action)
        mask = [(False if t[i] == env._max_episode_steps else done) for i, done in enumerate(dones)]

        self.buffer.append(state, action, reward, mask, log_pi, next_state)

        for i, done in enumerate(dones):
            if done:
                t[i] = 0
            # it seems we do not need to reset the env
            # because the env is automatically reset when done
            # next_state = env.reset()

        return next_state, t

    def update(self, writer):
        self.learning_steps += 1
        states, actions, rewards, dones, log_pis, next_states = \
            self.buffer.get()
        self.update_ppo(
            states, actions, rewards, dones, log_pis, next_states, writer)

    def update_ppo(self, states, actions, rewards, dones, log_pis, next_states,
                   writer):
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

        targets, gaes = calculate_gae(
            values, rewards, dones, next_values, self.gamma, self.lambd)

        for _ in range(self.epoch_ppo):
            for env_idx in range(rewards.size(1)):
                self.learning_steps_ppo += 1
                self.update_critic(
                    states[:, env_idx],
                    targets[:, env_idx],
                    writer
                )
                self.update_actor(
                    states[:, env_idx],
                    actions[:, env_idx],
                    log_pis[:, env_idx],
                    gaes[:, env_idx],
                    writer
                )

    def update_critic(self, states, targets, writer):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/critic', loss_critic.item(), self.learning_steps)

    def update_actor(self, states, actions, log_pis_old, gaes, writer):
        log_pis, dist_entropy = self.actor.evaluate(states, actions)
        entropy = dist_entropy.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()

        self.optim_actor.zero_grad()
        (loss_actor - self.coef_ent * entropy).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/actor', loss_actor.item(), self.learning_steps)
            writer.add_scalar(
                'stats/entropy', entropy.item(), self.learning_steps)

    def save_models(self, save_dir):
        super().save_models(save_dir)
        # We only save actor to reduce workloads.
        torch.save(
            self.actor.state_dict(),
            os.path.join(save_dir, 'actor.pth')
        )
