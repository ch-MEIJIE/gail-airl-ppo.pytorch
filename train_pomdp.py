import os
import argparse
from datetime import datetime
import torch

from gail_airl_ppo.algo.pomdp import pomdp
from gail_airl_ppo.trainer import Trainer
# from vec_env_wrapper import VecPyFlytEnvWrapper
from env_wrapper_async import PyFlytEnvWrapper


def run(args):
    num_env = 10
    env = PyFlytEnvWrapper(
        render_mode=None,
        env_id="PyFlyt/QuadX-Velocity-Gates-Asyn_v1",
        seed=args.seed
    )
    env_test = PyFlytEnvWrapper(
        render_mode=None,
        env_id="PyFlyt/QuadX-Velocity-Gates-Asyn_v1",
        seed=args.seed
    )

    action_dim = (env_test.act_size,)
    state_dims = (env.obs_atti_size+env.obs_target_size+env.obs_bound_size+1,)

    algo = pomdp(
        state_shape=state_dims,
        action_shape=action_dim,
        encoder='lstm',
        action_embedding_size=8,
        observ_embedding_size=32,
        reward_embedding_size=8,
        rnn_hidden_size=128,
        dqn_layers=[128, 128],
        policy_layers=[128, 128],
        # device=torch.device("cuda" if args.cuda else "cpu"),
        device=torch.device("cpu"),
        seed=args.seed,
        buffer_size=int(1e6),
        sampled_seq_len=64,
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, 'pomdp', f'seed{args.seed}-{time}')

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        num_env=num_env,
        seed=args.seed
    )
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num_steps', type=int, default=3*10**5)
    p.add_argument('--eval_interval', type=int, default=10000)
    p.add_argument('--env_id', type=str, default='QuadX-UVRZ-Gates-v2')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args)
