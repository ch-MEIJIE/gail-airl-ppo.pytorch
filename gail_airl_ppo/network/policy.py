import torch
from torch import nn
import torch.nn.functional as F
from ..torchkit.networks import Mlp
from .utils import build_mlp, reparameterize, evaluate_lop_pi


class StateIndependentPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states):
        return torch.tanh(self.net(states))

    def sample(self, states):
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states, actions):
        return evaluate_lop_pi(self.net(states), self.log_stds, actions)


class StateIndependentPolicyDiscrete(nn.Module):

    def __init__(
            self,
            state_shape,
            action_shape,
            hidden_units=(64, 64),
            hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states):
        return torch.softmax(self.net(states), dim=-1)

    def act(self, states):
        action_probs = self.forward(states)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def exploit(self, states):
        action_probs = self.forward(states)
        action = torch.argmax(action_probs, dim=-1)

        return action.detach()

    def evaluate(self, states, actions):
        action_probs = self.forward(states)
        dist = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        return action_logprobs, dist_entropy


class StateDependentPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=2 * action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states):
        return torch.tanh(self.net(states).chunk(2, dim=-1)[0])

    def sample(self, states):
        means, log_stds = self.net(states).chunk(2, dim=-1)
        return reparameterize(means, log_stds.clamp_(-20, 2))


class MarkovPolicyBase(Mlp):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_sizes,
        init_w=1e-3,
        image_encoder=None,
        **kwargs
    ):
        self.save_init_params(locals())
        self.action_dim = action_dim

        if image_encoder is None:
            self.input_size = obs_dim
        else:
            self.input_size = image_encoder.embed_size

        # first register MLP
        super().__init__(
            hidden_sizes,
            input_size=self.input_size,
            output_size=self.action_dim,
            init_w=init_w,
            **kwargs,
        )

        # then register image encoder
        self.image_encoder = image_encoder  # None or nn.Module

    def forward(self, obs):
        """
        :param obs: Observation, usually 2D (B, dim), but maybe 3D (T, B, dim)
        return action (*, dim)
        """
        x = self.preprocess(obs)
        return super().forward(x)

    def preprocess(self, obs):
        x = obs
        if self.image_encoder is not None:
            x = self.image_encoder(x)
        return x


class CategoricalPolicy(MarkovPolicyBase):
    """Based on https://github.com/ku2482/sac-discrete.pytorch/blob/master/sacd/model.py
    Usage: SAC-discrete
    ```
    policy = CategoricalPolicy(...)
    action, _, _ = policy(obs, deterministic=True)
    action, _, _ = policy(obs, deterministic=False)
    action, prob, log_prob = policy(obs, deterministic=False, return_log_prob=True)
    ```
    NOTE: action space must be discrete
    """
    PROB_MIN = 1e-8

    def forward(
        self,
        obs,
        deterministic=False,
        return_log_prob=False,
    ):
        """
        :param obs: Observation, usually 2D (B, dim), but maybe 3D (T, B, dim)
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        return: action (*, B, A), prob (*, B, A), log_prob (*, B, A)
        """
        action_logits = super().forward(obs)  # (*, A)

        prob, log_prob = None, None
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)  # (*)
            assert (
                return_log_prob is False
            )  # NOTE: cannot be used for estimating entropy
        else:
            prob = F.softmax(action_logits, dim=-1)  # (*, A)
            distr = torch.distributions.Categorical(prob)
            # categorical distr cannot reparameterize
            action = distr.sample()  # (*)
            if return_log_prob:
                log_prob = torch.log(torch.clamp(prob, min=self.PROB_MIN))

        # convert to one-hot vectors
        action = F.one_hot(action.long(), num_classes=self.action_dim).float()  # (*, A)

        return action, prob, log_prob