import torch
import torch.nn as nn
from torch.distributions import Categorical

# TODO: The Z axis can be removed from the attitude


class Backbone(nn.Module):

    def __init__(self, state_dim, hidden_unit_size, context_length):
        super(Backbone, self).__init__()
        self.attitude_dim = state_dim[0]
        self.attitude_net = nn.Sequential(
            nn.Linear(self.attitude_dim, hidden_unit_size),
            nn.ReLU(),
            nn.Linear(hidden_unit_size, hidden_unit_size),
            nn.ReLU(),
        )

        self.target_delta_dim = state_dim[1][0]*state_dim[1][1]
        self.target_num = state_dim[1][0]
        self.delta_dim = state_dim[1][1]
        self.target_font_net = nn.Sequential(
            nn.Linear(self.delta_dim, int(hidden_unit_size/2)),
            nn.ReLU(),
        )

        self.target_back_net = nn.Sequential(
            nn.Linear(hidden_unit_size, hidden_unit_size),
            nn.ReLU(),
        )

        self.bound_dim = state_dim[2]
        self.bounds_net = nn.Sequential(
            nn.Linear(self.bound_dim, hidden_unit_size),
            nn.ReLU(),
            nn.Linear(hidden_unit_size, hidden_unit_size),
            nn.ReLU(),
        )

        self.positional_encoding = nn.Parameter(
            torch.randn((context_length, int(hidden_unit_size/2))), requires_grad=True)

    def forward(self, state):
        # Operations on attitude
        obs_attitude = state[:self.attitude_dim]

        # Operations on target delta
        atti_output = self.attitude_net(obs_attitude)
        obs_target_delta = state[self.attitude_dim:
                                 self.attitude_dim+self.target_delta_dim]
        # reshape target delta
        obs_target_delta = obs_target_delta.reshape(
            self.target_num, self.delta_dim)
        obs_target_delta = obs_target_delta[..., :self.context_length, :]
        if len(obs_target_delta.shape) != len(self.positional_encoding.shape):
            # extend the positional encoding if the obs_target_delta has higer dimension
            pos_enc = torch.stack(
                [self.positional_encoding] * obs_target_delta.shape[0], dim=0)
        else:
            pos_enc = self.positional_encoding

        # Extract target delta features using the front net
        target_output = self.target_font_net(obs_target_delta)
        # Merge with positional encoding
        target_output = torch.cat((target_output, pos_enc), dim=-1)
        # Pass through the back net
        target_output = self.target_back_net(target_output)

        # mask then take mean
        mask = torch.ones_like(target_output, requires_grad=False)
        # mask out the padding
        mask[obs_target_delta.abs().sum(dim=-1) == 0] = 0.0
        target_output = target_output * mask
        target_output = target_output.mean(dim=-2)

        # Operations on target bounds
        obs_bounds = state[self.attitude_dim+self.target_delta_dim:]

        bound_output = self.bounds_net(obs_bounds)

        return atti_output, target_output, bound_output

    class Actor(nn.Module):

        def __init__(self, state_dim, action_dim, hidden_unit_size, context_length):
            super().__init__()
            self.action_dim = action_dim
            hidden_unit_size = hidden_unit_size
            self.backbone = Backbone(
                state_dim, hidden_unit_size, context_length)

            self.merge_net = nn.Sequential(
                nn.Linear(hidden_unit_size*3, hidden_unit_size),
                nn.ReLU(),
                nn.Linear(hidden_unit_size, hidden_unit_size),
                nn.ReLU(),
                nn.Linear(hidden_unit_size, action_dim),
                nn.Softmax(dim=-1)
            )

        def forward(self, state):
            atti_output, target_output, bound_output = self.backbone(state)
            merge_input = torch.cat(
                (atti_output, target_output, bound_output), dim=-1)
            action_probs = self.merge_net(merge_input)
            return action_probs

        def act(self, state):
            action_probs = self.forward(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)

            return action.detach(), action_logprob.detach()

        def exploit(self, state):
            action_probs = self.forward(state)
            action = torch.argmax(action_probs)

            return action.detach()

        def evaluate(self, state, action):
            action_probs = self.forward(state)
            dist = Categorical(action_probs)
            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()

            return action_logprobs, dist_entropy


class Critic(nn.Module):

    def __init__(self, state_dim, hidden_unit_size, context_length):
        super().__init__()
        self.backbone = Backbone(state_dim, hidden_unit_size, context_length)
        self.critic = nn.Sequential(
            nn.Linear(hidden_unit_size*3, hidden_unit_size),
            nn.ReLU(),
            nn.Linear(hidden_unit_size, hidden_unit_size),
            nn.ReLU(),
            nn.Linear(hidden_unit_size, 1)
        )

    def forward(self, state):
        atti_output, target_output, bound_output = self.backbone(state)
        merge_input = torch.cat(
            (atti_output, target_output, bound_output), dim=-1)
        value = self.critic(merge_input)
        return value

    def react(self, state):
        value = self.forward(state)
        return value

    def evaluate(self, state):
        value = self.forward(state)
        return value
