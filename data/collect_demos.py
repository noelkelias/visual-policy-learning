import numpy as np
import h5py


# =========================================================
# CORE DATA COLLECTION
# =========================================================
def collect_demos(env, policy, num_episodes=100, seed=0):

    observations = []
    states = []
    actions = []
    targets = []
    rewards = []
    dones = []
    success = []
    episode_ends = []

    np.random.seed(seed)

    step_counter = 0

    for ep in range(num_episodes):

        obs, info = env.reset(seed=seed + ep)

        done = False
        ep_steps = 0

        while not done:

            state = env.get_state()
            action = policy.predict(info)
            targets.append(info["target_pos"])

            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            # =================================================
            # STORE TRANSITION
            # =================================================
            observations.append(obs["image"])
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            success.append(info["success"])

            obs = next_obs

            step_counter += 1
            ep_steps += 1

        episode_ends.append(step_counter)

        print(f"[EP {ep}] steps={ep_steps} success={info['success']} dist={info['distance']:.3f}")

    dataset = {
        "observations": np.array(observations, dtype=np.uint8),
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "targets": np.array(targets, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "dones": np.array(dones, dtype=np.bool_),
        "success": np.array(success, dtype=np.bool_),
        "episode_ends": np.array(episode_ends, dtype=np.int32),
    }

    return dataset


# =========================================================
# HDF5 SAVER
# =========================================================
def save_hdf5(dataset, path):

    with h5py.File(path, "w") as f:

        for k, v in dataset.items():
            f.create_dataset(k, data=v, compression="gzip")

    print(f"Saved HDF5 dataset to: {path}")


# =========================================================
# HDF5 LOAD
# =========================================================
def load_hdf5(path):

    data = {}

    with h5py.File(path, "r") as f:
        for key in f.keys():
            data[key] = f[key][()]

    return data


# =========================================================
# HDF5 VERIFICATION
# =========================================================
def verify_dataset(dataset, num_samples=3):

    print("\n========== DATASET CHECK ==========")

    for k, v in dataset.items():
        print(f"{k}: shape={v.shape}, dtype={v.dtype}")

    print("\nSample checks:")

    for i in range(min(num_samples, len(dataset["actions"]))):

        print(f"\nStep {i}")
        print("action:", dataset["actions"][i])
        print("state:", dataset["states"][i][:5], "...")  # partial view
        print("reward:", dataset["rewards"][i])
        print("done:", dataset["dones"][i])
        print("success:", dataset["success"][i])

    # sanity checks
    assert dataset["observations"].shape[0] == dataset["actions"].shape[0]
    assert dataset["states"].shape[0] == dataset["actions"].shape[0]
    if "targets" in dataset:
        assert dataset["targets"].shape[0] == dataset["actions"].shape[0]

    print("\nDataset integrity check: PASSED")