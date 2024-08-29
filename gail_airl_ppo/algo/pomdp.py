import torch
import numpy as np
from copy import deepcopy
from torch.optim import Adam
from .sacd import SACD
from .base import Algorithm
from ..network.rnn_policy import Critic_RNN, Actor_RNN
from ..buffer import SeqReplayBuffer
import torch.nn.functional as F


class pomdp(Algorithm):

    def __init__(
            self,
            state_shape,
            action_shape,
            encoder,
            action_embedding_size,
            observ_embedding_size,
            reward_embedding_size,
            rnn_hidden_size,
            dqn_layers,
            policy_layers,
            device,
            seed,
            buffer_size,
            sampled_seq_len,
            batch_size=32,
            gamma=0.99,
            rnn_num_layers=1,
            lr=3e-4,
            tau=5e-3,
            image_encoder_fn=lambda: None,
            start_rollout=1  # TODO: Remeber to change this to 5
    ):
        super().__init__(state_shape, action_shape, device, seed, gamma)

        self.action_shape = action_shape
        self.state_shape = state_shape
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.algo = SACD(target_entropy=0.7, action_dim=action_shape[0])

        # Episode buffer
        self.buffer = SeqReplayBuffer(
            buffer_size,
            observation_dim=state_shape[0],
            action_dim=action_shape[0],
            sampled_seq_len=sampled_seq_len,
            sample_weight_baseline=0.0
        )

        # RNN Critic
        self.critic = Critic_RNN(
            obs_dim=state_shape[0],
            action_dim=action_shape[0],
            encoder=encoder,
            algo=self.algo,
            action_embedding_size=action_embedding_size,
            observ_embedding_size=observ_embedding_size,
            reward_embedding_size=reward_embedding_size,
            rnn_hidden_size=rnn_hidden_size,
            dqn_layers=dqn_layers,
            rnn_num_layers=rnn_num_layers,
            image_encoder=image_encoder_fn()
        )

        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        self.critic_target = deepcopy(self.critic)

        # RNN Actor
        self.actor = Actor_RNN(
            obs_dim=state_shape[0],
            action_dim=action_shape[0],
            encoder=encoder,
            algo=self.algo,
            action_embedding_size=action_embedding_size,
            observ_embedding_size=observ_embedding_size,
            reward_embedding_size=reward_embedding_size,
            rnn_hidden_size=rnn_hidden_size,
            policy_layers=policy_layers,
            rnn_num_layers=rnn_num_layers,
            image_encoder=image_encoder_fn()
        )

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        self.actor_target = deepcopy(self.actor)
        self.start_rollout = start_rollout
        self.current_rollout = 0

        # Initialize the hidden state
        self.prev_action, self.reward, self.internal_state = self.actor.get_initial_info()

        # Initial a temporary storage for the current episode
        self.initial_episode_storage()

        # do update flag
        self.do_update = False
        self.update_times = 0

    @torch.no_grad()
    def act(
        self,
        prev_internal_state,
        prev_action,
        reward,
        obs,
        deterministic=False,
        return_log_prob=False,
    ):
        prev_action = prev_action.unsqueeze(0)  # (1, B, dim)
        reward = reward.unsqueeze(0)  # (1, B, 1)
        obs = obs.unsqueeze(0)  # (1, B, dim)

        current_action_tuple, current_internal_state = self.actor.act(
            prev_internal_state=prev_internal_state,
            prev_action=prev_action,
            reward=reward,
            obs=obs,
            deterministic=deterministic,
            return_log_prob=return_log_prob,
        )
        return current_action_tuple, current_internal_state

    def explore(
        self,
        prev_internal_state,
        prev_action,
        reward,
        state
    ):
        (action, _, _, _), internal_state = self.act(
            prev_internal_state=prev_internal_state,
            prev_action=prev_action,
            reward=reward,
            obs=state,
            deterministic=False,
        )
        return action.cpu().detach().numpy(), internal_state

    def exploit(
        self,
        prev_internal_state,
        prev_action,
        reward,
        state
    ):
        (action, _, _, _), internal_state = self.act(
            prev_internal_state=prev_internal_state,
            prev_action=prev_action,
            reward=reward,
            obs=state,
            deterministic=True,
        )
        return action.cpu().detach().numpy(), internal_state

    def initial_episode_storage(self):
        self.state_list, self.action_list, self.reward_list, self.next_state_list, \
            self.done_list = [], [], [], [], []

    def step(self, env, state, t, step):
        # t is a numpy array of shape (E,)
        # E is the number of the parallel environments
        t += 1
        
        # Check if the state is a numpy array, if it is convert it to torch
        if not torch.is_tensor(state):
            state = torch.from_numpy(state).\
                view(-1, self.state_shape[0]).float().to(self.device)

        # select action
        if self.current_rollout <= self.start_rollout:
            # The action is not one-hot encoded
            action = env.action_space.sample()
        else:
            # The action is one-hot encoded
            action, self.internal_state = self.explore(
                self.internal_state, self.prev_action, self.reward, state
            )
            # find corresponding action from the one-hot encoded action
            action = np.argmax(action)

        # do action
        next_state, reward, done, info = env.step(action)

        # move to torch
        action = torch.FloatTensor([action]).to(self.device)
        # one-hot encode the action
        action = F.one_hot(action.to(torch.int64),
                           self.action_shape[0]).float()
        next_state = torch.from_numpy(next_state).\
            view(-1, self.state_shape[0]).float().to(self.device)
        reward = torch.FloatTensor([reward]).view(-1, 1).to(self.device)
        done = torch.from_numpy(np.array(done, dtype=int)
                                ).view(-1, 1).to(self.device)

        # Check if the episode is done
        done_rollout = False if done[0][0].to(
            "cpu").detach().numpy() == 0.0 else True

        # Ignore the time limit
        term = (
            False
            if "TimeLimit.truncated" in info or t >= env.max_steps
            else done_rollout
        )

        self.state_list.append(state)
        self.action_list.append(action)
        self.reward_list.append(reward)
        self.next_state_list.append(next_state)
        self.done_list.append(term)

        if done_rollout:
            self.current_rollout += 1
            if self.current_rollout > self.start_rollout:
                self.do_update = True
            # Record the t
            self.update_times = t
            # Store the episode
            self.buffer.add_episode(
                observations=torch.cat(self.state_list, dim=0),
                actions=torch.cat(self.action_list, dim=0),
                rewards=torch.cat(self.reward_list, dim=0),
                terminals=torch.from_numpy(
                    np.array(self.done_list, dtype=int).reshape(-1, 1)).to(self.device),
                next_observations=torch.cat(self.next_state_list, dim=0)
            )
            self.initial_episode_storage()
            # reset the environment
            next_state = env.reset()
            # convert the next_state to torch
            next_state = torch.from_numpy(next_state).\
                view(-1, self.state_shape[0]).float().to(self.device)
            # clear t
            t = 0

        return next_state, t

    def is_update(self, step):
        return self.do_update

    def update(self, writer):
        print("Update")
        # Sample a batch
        for _ in range(self.update_times):
            batch = self.buffer.random_episodes(self.batch_size)
            self.update_once(batch, writer)

        # reset the update flag
        self.do_update = False

    def update_once(self, batch, writer):
        # unpack the batch
        actions, rewards, dones = batch["act"], batch["rew"], batch["term"]
        _, batch_size, _ = actions.shape
        masks = batch["mask"]
        obs, next_obs = batch["obs"], batch["obs2"]  # (T, B, dim)

        # extend observs, actions, rewards, dones from len = T to len = T+1
        observs = torch.cat((obs[[0]], next_obs), dim=0)  # (T+1, B, dim)
        actions = torch.cat(
            (
                torch.zeros((1, batch_size, self.action_shape[0])
                            ).float().to(self.device),
                actions
            ),
            dim=0
        )  # (T+1, B, dim)
        rewards = torch.cat(
            (torch.zeros((1, batch_size, 1)).float().to(self.device), rewards), dim=0
        )  # (T+1, B, dim)
        dones = torch.cat(
            (torch.zeros((1, batch_size, 1)).float().to(self.device), dones), dim=0
        )  # (T+1, B, dim)

        # Check
        assert (
            actions.dim()
            == rewards.dim()
            == dones.dim()
            == observs.dim()
            == masks.dim()
            == 3
        )
        assert (
            actions.shape[0]
            == rewards.shape[0]
            == dones.shape[0]
            == observs.shape[0]
            == masks.shape[0] + 1
        )

        num_valid = torch.clamp(masks.sum(), min=1.0)

        self.update_critic(observs, actions, rewards,
                           dones, num_valid, masks, writer)
        log_probs = self.update_actor(
            observs, actions, rewards, num_valid, masks, writer)
        self.update_target()
        self.update_others(log_probs, masks, num_valid)

    def update_critic(self, observs, actions, rewards, dones, num_valid, masks, writer):
        (q1_pred, q2_pred), q_target = self.algo.critic_loss(
            markov_actor=False,
            markov_critic=False,
            actor=self.actor,
            actor_target=self.actor_target,
            critic=self.critic,
            critic_target=self.critic_target,
            observs=observs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            gamma=self.gamma,
        )

        q1_pred, q2_pred = q1_pred * masks, q2_pred * masks
        q_target = q_target * masks
        qf1_loss = ((q1_pred - q_target) ** 2).sum() / num_valid  # TD error
        qf2_loss = ((q2_pred - q_target) ** 2).sum() / num_valid  # TD error

        self.critic_optimizer.zero_grad()
        (qf1_loss + qf2_loss).backward()
        self.critic_optimizer.step()

    def update_actor(self, observs, actions, rewards, num_valid, masks, writer):
        policy_loss, log_probs = self.algo.actor_loss(
            markov_actor=False,
            markov_critic=False,
            actor=self.actor,
            actor_target=self.actor_target,
            critic=self.critic,
            critic_target=self.critic_target,
            observs=observs,
            actions=actions,
            rewards=rewards,
        )

        policy_loss = (policy_loss * masks).sum() / num_valid

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        return log_probs

    def update_target(self):
        self.soft_update_from_to(self.critic, self.critic_target, self.tau)

    def update_others(self, log_probs, masks, num_valid):
        if log_probs is not None:
            with torch.no_grad():
                current_log_probs = (log_probs[:-1] * masks).sum() / num_valid
                current_log_probs = current_log_probs.item()
            _ = self.algo.update_others(current_log_probs)

    @staticmethod
    def soft_update_from_to(source, target, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau)
    
    def evaluate(self, env_test):
        obs = env_test.reset()
        done = False
        # convert the obs to torch
        obs = torch.from_numpy(obs).view(-1, self.state_shape[0]).float().to(self.device)
        # get hideen state from the actor
        action, reward, internal_state = self.actor.get_initial_info()
        epsidoic_return = 0.0
        while not done:
            action, internal_state = self.exploit(internal_state, action, reward, obs)
            action = np.argmax(action)
            obs, reward, done, _ = env_test.step(action)
            obs = torch.from_numpy(obs).view(-1, self.state_shape[0]).float().to(self.device)
            epsidoic_return += reward
            reward = torch.FloatTensor([reward]).view(-1, 1).to(self.device)
            action = torch.FloatTensor([action]).to(self.device)
            # one-hot encode the action
            action = F.one_hot(action.to(torch.int64),
                               self.action_shape[0]).float()
        return epsidoic_return

    def save_models(self, save_dir):
        super().save_models(save_dir)
        torch.save(
            self.critic.state_dict(),
            f"{save_dir}/critic.pth"
        )
        torch.save(
            self.actor.state_dict(),
            f"{save_dir}/actor.pth"
        )
