# policies/sac_policy.py
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from training.rl_utils import scale_action, unscale_action

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


# =========================
# NETWORK
# =========================
def mlp(in_dim, out_dim, hidden=256):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim, hidden)

    def forward(self, obs):
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self(obs)
        dist = Normal(mean, log_std.exp())
        x = dist.rsample()
        action = torch.tanh(x)
        log_prob = dist.log_prob(x) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1, hidden)
        self.q2 = mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)



# =========================
# REPLAY
# =========================
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, device):
        self.device = device
        self.size = size
        self.ptr = 0
        self.full = False
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts = np.zeros((size, act_dim), dtype=np.float32)
        self.rews = np.zeros((size, 1), dtype=np.float32)
        self.dones = np.zeros((size, 1), dtype=np.float32)

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.size
        self.full = self.full or self.ptr == 0

    def __len__(self):
        return self.size if self.full else self.ptr

    def sample(self, batch_size):
        idx = np.random.randint(0, len(self), size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], device=self.device),
            torch.as_tensor(self.acts[idx], device=self.device),
            torch.as_tensor(self.rews[idx], device=self.device),
            torch.as_tensor(self.next_obs[idx], device=self.device),
            torch.as_tensor(self.dones[idx], device=self.device),
        )


def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)



# =========================
# POLICY
# =========================
class SACPolicy:
    """SAC. Actions in [-1, 1]; unscaled on env.step."""

    def __init__(
        self,
        obs_dim,
        act_dim,
        act_low,
        act_high,
        hidden=256,
        device="cpu",
        actor=None,
        critic=None,
        log_alpha=None,
    ):
        self.device = torch.device(device)
        self.act_low = np.asarray(act_low, dtype=np.float32)
        self.act_high = np.asarray(act_high, dtype=np.float32)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden

        self.actor = (actor or Actor(obs_dim, act_dim, hidden)).to(self.device)
        self.critic = (critic or Critic(obs_dim, act_dim, hidden)).to(self.device)
        self.critic_target = Critic(obs_dim, act_dim, hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        if log_alpha is None:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        else:
            self.log_alpha = log_alpha.detach().clone().to(self.device).requires_grad_(True)

    def predict(self, obs, deterministic=True):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                _, _, scaled = self.actor.sample(obs_t)
            else:
                scaled, _, _ = self.actor.sample(obs_t)
            scaled = scaled.cpu().numpy()[0]
        return unscale_action(scaled, self.act_low, self.act_high)

    def save(self, path):
        path = path if path.endswith(".pt") else path + ".pt"
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
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
        tau=0.005,
        buffer_size=100_000,
        batch_size=256,
        learning_starts=100,
        updates_per_step=1,
        seed=42,
        curve_path=None,
        log_every=2000,
        max_steps=200,
    ):
        actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        target_entropy = -float(self.act_dim)
        buf = ReplayBuffer(self.obs_dim, self.act_dim, buffer_size, self.device)

        np.random.seed(seed)
        torch.manual_seed(seed)
        obs, _ = env.reset(seed=seed)
        ep_ret, ep_len = 0.0, 0
        t0 = time.time()

        if curve_path:
            os.makedirs(os.path.dirname(curve_path) or ".", exist_ok=True)
            with open(curve_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["timesteps", "episode_reward", "episode_length", "success"]
                )

        for t in range(1, total_timesteps + 1):
            if t < learning_starts:
                env_action = env.action_space.sample().astype(np.float32)
                scaled_action = scale_action(env_action, self.act_low, self.act_high)
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(
                        obs, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    scaled_t, _, _ = self.actor.sample(obs_t)
                    scaled_action = scaled_t.cpu().numpy()[0]
                env_action = unscale_action(scaled_action, self.act_low, self.act_high)

            next_obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            buf.add(obs, scaled_action, reward, next_obs, terminated)
            obs = next_obs
            ep_ret += reward
            ep_len += 1

            if done:
                if curve_path:
                    with open(curve_path, "a", newline="") as f:
                        csv.writer(f).writerow(
                            [
                                t,
                                ep_ret,
                                ep_len,
                                int(bool(info.get("success", False))),
                            ]
                        )
                if t % log_every < max_steps:
                    print(
                        f"t={t:6d}  ep_ret={ep_ret:8.2f}  len={ep_len:3d}  "
                        f"success={int(bool(info.get('success', False)))}  "
                        f"alpha={self.log_alpha.exp().item():.3f}"
                    )
                obs, _ = env.reset()
                ep_ret, ep_len = 0.0, 0

            if t < learning_starts or len(buf) < batch_size:
                continue

            for _ in range(updates_per_step):
                o, a, r, no, d = buf.sample(batch_size)
                alpha = self.log_alpha.exp().detach()

                with torch.no_grad():
                    next_a, next_logp, _ = self.actor.sample(no)
                    q1_t, q2_t = self.critic_target(no, next_a)
                    q_t = torch.min(q1_t, q2_t) - alpha * next_logp
                    backup = r + gamma * (1.0 - d) * q_t

                q1, q2 = self.critic(o, a)
                critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()

                a_pi, logp, _ = self.actor.sample(o)
                q1_pi, q2_pi = self.critic(o, a_pi)
                q_pi = torch.min(q1_pi, q2_pi)
                actor_loss = (alpha * logp - q_pi).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()

                alpha_loss = -(self.log_alpha * (logp + target_entropy).detach()).mean()
                alpha_opt.zero_grad()
                alpha_loss.backward()
                alpha_opt.step()

                soft_update(self.critic_target, self.critic, tau)

        print(f"SAC wall time: {time.time() - t0:.1f}s")
        return self


def load_sac_policy(path, device="cpu"):
    path = path if path.endswith(".pt") else path + ".pt"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    policy = SACPolicy(
        obs_dim=int(ckpt["obs_dim"]),
        act_dim=int(ckpt["act_dim"]),
        act_low=ckpt["act_low"],
        act_high=ckpt["act_high"],
        hidden=int(ckpt.get("hidden", 256)),
        device=device,
    )
    policy.actor.load_state_dict(ckpt["actor"])
    policy.critic.load_state_dict(ckpt["critic"])
    policy.critic_target.load_state_dict(ckpt["critic"])
    policy.log_alpha = ckpt["log_alpha"].to(policy.device).requires_grad_(True)
    return policy
