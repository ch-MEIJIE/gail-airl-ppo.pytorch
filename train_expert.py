import os
import argparse
from datetime import datetime
import torch

from gail_airl_ppo.algo.ppo_discrete import PPO
from gail_airl_ppo.trainer import Trainer
from env_wrapper import PyFlytEnvWrapper


def run(args):
    env = PyFlytEnvWrapper(
        render_mode=None,
        env_id="PyFlyt/QuadX-UVRZ-Gates-v2"
    )
    env_test = PyFlytEnvWrapper(
        render_mode=None,
        env_id="PyFlyt/QuadX-UVRZ-Gates-v2"
    )
    
    action_dim = env.act_size
    state_dims = [env.obs_atti_size, (env.targets_num, env.obs_target_size), env.obs_bound_size]

    algo = PPO(
        state_shape=state_dims,
        action_shape=action_dim,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, 'PPO', f'seed{args.seed}-{time}')

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed
    )
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num_steps', type=int, default=10**6)
    p.add_argument('--eval_interval', type=int, default=10**4)
    p.add_argument('--env_id', type=str, default='QuadX-UVRZ-Gates-v2')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args)
