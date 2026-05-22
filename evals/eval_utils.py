import os
import numpy as np
import torch
import imageio
import torch.nn.functional as F
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess_image(obs_image, resize_to=None):
    """HWC uint8 -> NCHW float in [0, 1]. Encoders resize/normalize internally."""
    img = torch.from_numpy(obs_image).to(DEVICE, dtype=torch.float32) / 255.0

    if img.ndim == 3:
        img = img.permute(2, 0, 1)

    img = img.unsqueeze(0)

    if resize_to is not None:
        img = F.interpolate(
            img,
            size=(resize_to, resize_to),
            mode="bilinear",
            align_corners=False,
        )

    return img


def extract_image(obs):
    return obs["image"] if isinstance(obs, dict) else obs


def reset_env(env):
    out = env.reset()
    if isinstance(out, tuple):
        return out[0], out[1] if len(out) > 1 else {}
    return out, {}


def step_env(env, action):
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        return out
    obs, reward, done, info = out
    return obs, reward, done, False, info


# -----------------------------
# EPISODE ROLLOUT
# -----------------------------
def rollout_episode(
    env,
    policy,
    max_steps=100,
    save_video=False,
    video_path=None,
    render_every=2,
    use_amp=True,
    action_repeat=1,
):

    obs, info = reset_env(env)
    policy.eval()

    total_reward = 0.0
    success = 0
    frames = []
    action = None

    for step in range(max_steps):

        if action is None or (step % action_repeat) == 0:
            img = extract_image(obs)
            inp = preprocess_image(img)

            with torch.inference_mode(), torch.amp.autocast(
                "cuda", enabled=(use_amp and DEVICE == "cuda")
            ):
                action = policy(inp)

            action = action.squeeze(0).float().cpu().numpy()
            action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)

            if hasattr(env, "action_space"):
                low = np.asarray(env.action_space.low)
                high = np.asarray(env.action_space.high)
                action = np.clip(action, low, high)

        obs, reward, terminated, truncated, info = step_env(env, action)

        total_reward += float(reward)

        if save_video and (step % render_every == 0):
            frames.append(extract_image(obs))

        if isinstance(info, dict) and info.get("success", False):
            success = 1

        if terminated or truncated:
            break

    if save_video and video_path and frames:
        imageio.mimsave(video_path, frames, fps=20)

    return {
        "success": success,
        "reward": float(total_reward),
        "steps": step + 1,
    }


# -----------------------------
# EVAL LOOP (FAST)
# -----------------------------
def evaluate_policy(
    env_fn,
    policy,
    num_episodes=10,
    max_steps=100,
    action_repeat=1,
    use_amp=True,
    close_env=True,
):

    policy.eval()

    env = env_fn()   # reuse env

    successes, rewards, steps_list = [], [], []

    for _ in tqdm(range(num_episodes)):

        metrics = rollout_episode(
            env=env,
            policy=policy,
            max_steps=max_steps,
            save_video=False,
            action_repeat=action_repeat,
            use_amp=use_amp,
        )

        successes.append(metrics["success"])
        rewards.append(metrics["reward"])
        steps_list.append(metrics["steps"])

    if close_env:
        env.close()

    return {
        "success_rate": float(np.mean(successes)),
        "avg_reward": float(np.mean(rewards)),
        "avg_steps": float(np.mean(steps_list)),
        "reward_std": float(np.std(rewards)),
    }