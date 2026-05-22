import h5py
import numpy as np
import imageio

# your mujoco env
from envs.panda_env import PandaEnv  # adjust path

def render_dataset(input_path, output_path):
    env = PandaEnv(render_mode="rgb_array")

    with h5py.File(input_path, "r") as f:
        states = f["states"][:]   # must exist!
        actions = f["actions"][:]

    images = []

    for s in states:
        env.set_state(s)          # IMPORTANT: deterministic replay
        img = env.render()        # (H, W, 3)
        images.append(img)

    images = np.array(images, dtype=np.uint8)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("observations", data=images)
        f.create_dataset("actions", data=actions)

    print("Saved dataset with images:", images.shape)


if __name__ == "__main__":
    render_dataset(
        "data/panda_demos_raw.h5",
        "data/panda_demos.h5"
    )