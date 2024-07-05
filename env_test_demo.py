from vec_env_wrapper import PyFlytEnvWrapper


if __name__ == '__main__':
    env = PyFlytEnvWrapper(
        render_mode=None,
        env_id="PyFlyt/QuadX-UVRZ-Gates-v2",
        num_env=5
    )

    obs = env.reset()
    pass