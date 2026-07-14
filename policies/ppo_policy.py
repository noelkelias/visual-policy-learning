# policies/ppo_policy.py

import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from training.rl_utils import ObsRewardNormalizer, unscale_action


# =========================
# NETWORK
# =========================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):

    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)

    return layer


class ActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden=64):

        super().__init__()

        self.actor_body = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )

    def value(self, obs):
        return self.critic(obs)

    def get_action_and_value(self, obs, action=None):

        mean = self.actor_body(obs)
        std = self.actor_logstd.expand_as(mean).exp()
        dist = Normal(mean, std)

        if action is None:

            x = dist.rsample()
            action = torch.tanh(x)
            log_prob = dist.log_prob(x) - torch.log(1.0 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(-1)

        else:

            x = torch.atanh(action.clamp(-0.999999, 0.999999))
            log_prob = dist.log_prob(x) - torch.log(1.0 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(-1)

        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)

        return action, log_prob, entropy, value

    def act_mean(self, obs):
        return torch.tanh(self.actor_body(obs))


# =========================
# GAE
# =========================
def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):

    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(n)):

        next_nonterminal = 1.0 - float(dones[t])
        next_v = next_value if t == n - 1 else values[t + 1]

        delta = rewards[t] + gamma * next_v * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae

    return advantages, advantages + values


# =========================
# POLICY
# =========================
class PPOPolicy:
    """PPO. Actions in [-1, 1]; unscaled on env.step."""

    def __init__(
        self,
        obs_dim,
        act_dim,
        act_low,
        act_high,
        hidden=64,
        device="cpu",
        gamma=0.99,
        ac=None,
        normalizer=None,
    ):

        self.device = torch.device(device)
        self.act_low = np.asarray(act_low, dtype=np.float32)
        self.act_high = np.asarray(act_high, dtype=np.float32)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden

        self.ac = (ac or ActorCritic(obs_dim, act_dim, hidden)).to(self.device)
        self.normalizer = normalizer or ObsRewardNormalizer((obs_dim,), gamma=gamma)

    def predict(self, obs, deterministic=True):
        obs_n = self.normalizer.normalize_obs(obs, update=False)
        obs_t = torch.as_tensor(obs_n, dtype=torch.float32, device=self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                scaled = self.ac.act_mean(obs_t).cpu().numpy()[0]
            else:
                scaled, _, _, _ = self.ac.get_action_and_value(obs_t)
                scaled = scaled.cpu().numpy()[0]
        return unscale_action(scaled, self.act_low, self.act_high)

    def save(self, path):
        path = path if path.endswith(".pt") else path + ".pt"
        torch.save(
            {
                "ac": self.ac.state_dict(),
                "normalizer": self.normalizer.state_dict(),
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "act_low": self.act_low,
                "act_high": self.act_high,
                "hidden": self.hidden,
            },
            path,
        )
        return path

    def learn(
        self,
        env,
        total_timesteps,
        *,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        seed=42,
        curve_path=None,
        log_every=5000,
        max_steps=200,
    ):
        optimizer = torch.optim.Adam(self.ac.parameters(), lr=lr, eps=1e-5)
        np.random.seed(seed)
        torch.manual_seed(seed)

        obs, _ = env.reset(seed=seed)
        obs = self.normalizer.normalize_obs(obs, update=True)
        global_step = 0
        ep_ret_raw, ep_len = 0.0, 0
        t0 = time.time()
        num_updates = max(1, total_timesteps // n_steps)

        if curve_path:
            os.makedirs(os.path.dirname(curve_path) or ".", exist_ok=True)
            with open(curve_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["timesteps", "episode_reward", "episode_length", "success"]
                )

        for update in range(1, num_updates + 1):
            obs_buf = np.zeros((n_steps, self.obs_dim), dtype=np.float32)
            act_buf = np.zeros((n_steps, self.act_dim), dtype=np.float32)
            logp_buf = np.zeros(n_steps, dtype=np.float32)
            rew_buf = np.zeros(n_steps, dtype=np.float32)
            done_buf = np.zeros(n_steps, dtype=np.float32)
            val_buf = np.zeros(n_steps, dtype=np.float32)

            for step in range(n_steps):
                global_step += 1
                obs_buf[step] = obs
                with torch.no_grad():
                    obs_t = torch.as_tensor(
                        obs, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    action, logp, _, value = self.ac.get_action_and_value(obs_t)
                    action_np = action.cpu().numpy()[0]
                    logp_buf[step] = logp.cpu().numpy()[0]
                    val_buf[step] = value.cpu().numpy()[0]

                act_buf[step] = action_np
                env_action = unscale_action(action_np, self.act_low, self.act_high)
                next_obs_raw, reward_raw, terminated, truncated, info = env.step(
                    env_action
                )
                done = terminated or truncated

                reward_for_learn = float(reward_raw)
                if truncated and not terminated:
                    with torch.no_grad():
                        next_n = self.normalizer.normalize_obs(
                            next_obs_raw, update=False
                        )
                        next_t = torch.as_tensor(
                            next_n, dtype=torch.float32, device=self.device
                        ).unsqueeze(0)
                        reward_for_learn += gamma * float(
                            self.ac.value(next_t).reshape(-1)[0].cpu().item()
                        )

                rew_buf[step] = self.normalizer.normalize_reward(
                    reward_for_learn, update=True
                )
                done_buf[step] = float(done)
                ep_ret_raw += float(reward_raw)
                ep_len += 1

                if done:
                    if curve_path:
                        with open(curve_path, "a", newline="") as f:
                            csv.writer(f).writerow(
                                [
                                    global_step,
                                    ep_ret_raw,
                                    ep_len,
                                    int(bool(info.get("success", False))),
                                ]
                            )
                    if global_step % log_every < max_steps:
                        print(
                            f"t={global_step:6d}  ep_ret={ep_ret_raw:8.2f}  "
                            f"len={ep_len:3d}  "
                            f"success={int(bool(info.get('success', False)))}"
                        )
                    next_obs_raw, _ = env.reset()
                    self.normalizer.reset_ret()
                    ep_ret_raw, ep_len = 0.0, 0

                obs = self.normalizer.normalize_obs(next_obs_raw, update=True)

            with torch.no_grad():
                obs_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                next_value = float(self.ac.value(obs_t).cpu().numpy()[0, 0])

            advantages, returns = compute_gae(
                rew_buf, val_buf, done_buf, next_value, gamma, gae_lambda
            )

            b_obs = torch.as_tensor(obs_buf, device=self.device)
            b_act = torch.as_tensor(act_buf, device=self.device)
            b_logp = torch.as_tensor(logp_buf, device=self.device)
            b_adv = torch.as_tensor(advantages, device=self.device)
            b_ret = torch.as_tensor(returns, device=self.device)
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            inds = np.arange(n_steps)
            for _ in range(n_epochs):
                np.random.shuffle(inds)
                for start in range(0, n_steps, batch_size):
                    mb = inds[start : start + batch_size]
                    _, new_logp, entropy, new_value = self.ac.get_action_and_value(
                        b_obs[mb], b_act[mb]
                    )
                    ratio = (new_logp - b_logp[mb]).exp()
                    pg1 = -b_adv[mb] * ratio
                    pg2 = -b_adv[mb] * torch.clamp(
                        ratio, 1.0 - clip_range, 1.0 + clip_range
                    )
                    loss = (
                        torch.max(pg1, pg2).mean()
                        - ent_coef * entropy.mean()
                        + vf_coef * 0.5 * ((new_value - b_ret[mb]) ** 2).mean()
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.parameters(), max_grad_norm)
                    optimizer.step()

            if update == num_updates or (update * n_steps) % max(n_steps, 25000) == 0:
                print(f"[ppo] update={update}/{num_updates} steps={global_step}")

        print(f"PPO wall time: {time.time() - t0:.1f}s")
        return self


def load_ppo_policy(path, device="cpu"):
    path = path if path.endswith(".pt") else path + ".pt"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    policy = PPOPolicy(
        obs_dim=int(ckpt["obs_dim"]),
        act_dim=int(ckpt["act_dim"]),
        act_low=ckpt["act_low"],
        act_high=ckpt["act_high"],
        hidden=int(ckpt.get("hidden", 64)),
        device=device,
    )
    policy.ac.load_state_dict(ckpt["ac"])
    if "normalizer" in ckpt:
        policy.normalizer.load_state_dict(ckpt["normalizer"])
    else:
        policy.normalizer.obs_rms.mean = np.asarray(ckpt["obs_mean"], dtype=np.float64)
        policy.normalizer.obs_rms.var = np.asarray(ckpt["obs_var"], dtype=np.float64)
    return policy
