from evals.eval_utils import evaluate_policy
from evals.wrappers import (
    LightingWrapper,
    OcclusionWrapper,
    GaussianNoiseWrapper,
)


def run_robustness_suite(env_fn, models, num_episodes=20):

    tests = {
        "normal": lambda env: env,
        "dark": lambda env: LightingWrapper(env, brightness=0.5),
        "bright": lambda env: LightingWrapper(env, brightness=1.5),
        "occlusion": lambda env: OcclusionWrapper(env),
        "noise": lambda env: GaussianNoiseWrapper(env, noise_std=15),
    }

    results = {}

    for test_name, wrapper_fn in tests.items():

        print(f"\n=== {test_name} ===")
        results[test_name] = {}

        # create env ONCE per condition (huge speedup)
        base_env = wrapper_fn(env_fn())

        for model_name, model in models.items():

            model.eval()

            metrics = evaluate_policy(
                env_fn=lambda: base_env,
                policy=model,
                num_episodes=num_episodes,
                close_env=False,
            )

            results[test_name][model_name] = metrics
            print(model_name, metrics)

        base_env.close()

    return results