from torch.nn import functional as F

variant = dict(
    env_kwargs=dict(),
    imsize=84,
    rl_variant=dict(
        algo_kwargs=dict(
            base_kwargs=dict(
                num_epochs=300,
                num_steps_per_epoch=3000,
                batch_size=128,
                num_rollouts_per_eval=1,
            ),
            td3_kwargs=dict(),
            twin_sac_kwargs=dict(),
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_are_rollout_goals=0.2,
            fraction_resampled_goals_are_env_goals=0.25,
            fraction_resampled_goals_are_replay_buffer_goals=0.25,
        ),
        exploration_noise=0.1,
        exploration_type='epsilon',
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
            hidden_activation=F.relu,
        ),
        vf_kwargs=dict(
            hidden_sizes=[400, 300],
            hidden_activation=F.relu,
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
            hidden_activation=F.relu,
        ),
        algorithm="TD3",
        vis_kwargs=dict(
            save_video=True,
            num_samples_for_video=10,
        ),
        SubgoalPlanner_kwargs=dict(
            realistic_subgoal_weight=0.0,
            use_realistic_hard_constraint=False,
            realistic_hard_constraint_threshold=1.75,
            cost_mode='min',
            replan_freq=-1,
            optimizer='cem',
            optimize_over_states=False,
            use_true_prior_for_init=True,
            cem_optimizer_kwargs=dict(
                batch_size=1000,
                frac_top_chosen=0.05,
                num_iters=15,
                use_init_subgoals=True,
            ),
            gradient_optimizer_kwargs=dict(
                lr=1e-3,
            )
        ),
    ),
    train_reprojection_network_variant=dict(
        use_cached_network=True,
        num_epochs=500,  # 1000,
        generate_reprojection_network_dataset_kwargs=dict(
            N=int(1e6),
            test_p=0.9,
        ),
        reprojection_network_kwargs=dict(),
        algo_kwargs=dict(
            lr=1e-3,
            batch_size=int(2 ** 16),
        ),
    ),
)