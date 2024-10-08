import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from src.env.inertia_env import InertiaEnv
from src.models.market_model import calculate_market_share
from ..config import CUSTOMERS, PERIODS, STARTING_PRICES, LARGEST_DISCOUNT, LARGEST_INCREASE, NUM_AGENTS, TIMESTEPS_TOTAL, MAX_TRAINING_ITERATION

if ray.is_initialized():
    ray.shutdown()
ray.init(ignore_reinit_error=True)

# Define training parameters
config = (
    PPOConfig()
    .environment(InertiaEnv)
    .framework('torch')
    .training(train_batch_size=5000)
    .resources(num_gpus=0)
    .debugging(seed=0)
    .rollouts(num_rollout_workers=10, num_envs_per_worker=5, rollout_fragment_length="auto")
)

stop = {
    "timesteps_total": TIMESTEPS_TOTAL,
    "training_iteration": MAX_TRAINING_ITERATION,
}

results = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=air.RunConfig(
        storage_path="",
        stop=stop,
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=5,
            checkpoint_at_end=True)
    ),
).fit()

ray.shutdown()