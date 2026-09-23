"""mappo.py — MAPPO trainer for the MATE camera team.

Design mirrors the MATE example (RLlib-based) but is self-contained torch:
  * shared Gaussian policy across cameras (parameter sharing),
  * centralized critic on the concatenated team observation,
  * GAE(lambda) advantages, clipped PPO surrogate, entropy bonus,
  * frame_skip: one action applied for several env steps (MATE default 5),
  * checkpointing: latest.pt every interval, best.pt by eval return,
    resume-from-checkpoint (Phase 3 gate).
"""

import os

import numpy as np
import torch

from .models import CentralizedCritic, MLPPolicy, ObsNormalizer


class RolloutBuffer:
    """Simple list-based rollout buffer for a team trajectory."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.obs = []            # (batch, n_cam, obs_dim)
        self.actions = []        # (batch, n_cam, act_dim)
        self.rewards = []
        self.values = []
        self.dones = []
        self.log_probs = []
        self.masks = []          # next-done handling for GAE

    def add(self, obs, action, reward, value, done, log_prob):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(bool(done))
        self.log_probs.append(log_prob)

    def finalize(self, last_value, done):
        """Compute GAE advantages/returns. Returns tuple of stacked tensors.

        GAE over the collected segment: terminal transition bootstraps with
        `last_value` only when the episode did not end inside the segment.
        """
        obs = np.asarray(self.obs, dtype=np.float32)
        actions = np.asarray(self.actions, dtype=np.float32)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=bool)

        n = len(rewards)
        if n == 0:
            return (torch.zeros((0,), dtype=torch.float32),) * 5

        adv = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            next_val = last_value if t == n - 1 else values[t + 1]
            next_done = dones[-1] if t == n - 1 else dones[t + 1]
            delta = rewards[t] + self.gamma * next_val * (not next_done) - values[t]
            gae = delta + self.gamma * self.lam * (not next_done) * gae
            adv[t] = gae
        returns = adv + values

        return (
            torch.as_tensor(obs),
            torch.as_tensor(actions),
            torch.as_tensor(rewards),
            torch.as_tensor(adv),
            torch.as_tensor(returns),
        )

    def __len__(self):
        return len(self.rewards)


class MAPPO:
    def __init__(self, obs_dim, act_dim, n_cam, seed=None,
                 hidden=(512, 256), lr=5e-4, gamma=0.99, lam=0.95,
                 clip=0.2, epochs=10, minibatch=128, entropy_coef=0.0,
                 value_coef=0.5, frame_skip=5, device="cpu",
                 logger=None, log_every=50):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_cam = n_cam
        self.device = torch.device(device)

        rng = np.random.default_rng(seed)
        torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
        self.seed = seed

        self.obs_normalizer = ObsNormalizer(obs_dim)
        self.policy = MLPPolicy(obs_dim, act_dim, hidden=hidden,
                                normalizer=self.obs_normalizer)
        self.critic = CentralizedCritic(n_cam, obs_dim, hidden=hidden,
                                        normalizer=self.obs_normalizer)

        self.opt = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )

        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.epochs = epochs
        self.minibatch = minibatch
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.frame_skip = frame_skip
        self.logger = logger
        self.log_every = log_every

        self.step_count = 0
        self.episode_count = 0
        self.last_ep_returns = []
        self._buffer = RolloutBuffer()
        self._buffer.gamma = self.gamma
        self._buffer.lam = self.lam

    # ------------------------------------------------------------------
    def fit_obs_normalizer(self, env, samples=512, seed=None):
        """Sample observations with a random policy and fit the per-dimension
        normalizer (identity until called). Must be consistent across acting,
        critic and update — it is, because normalization lives in the models."""
        if self.obs_normalizer.fitted:
            return self.obs_normalizer
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        obs = env.reset()
        n_cam, obs_dim = obs.shape
        collected = np.empty((0, obs_dim), dtype=np.float32)
        done = False
        steps = 0
        while len(collected) < samples and not done and steps < samples:
            actions = rng.uniform(-1.0, 1.0, size=(n_cam, 2))
            obs_next, _, done, _ = env.step(actions)
            collected = np.concatenate([collected, obs], axis=0)
            obs = obs_next
            steps += 1
        self.obs_normalizer.fit(collected)
        return self.obs_normalizer

    def act_batch(self, obs, deterministic=False):
        """obs: (n_cam, obs_dim) -> (actions (n_cam, act_dim) clipped to [-1,1],
        joint log-prob (scalar = sum over cameras))."""
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        with torch.no_grad():
            dist = self.policy.get_distribution(obs_t)
            a = dist.mean if deterministic else dist.sample()
            logp = dist.log_prob(a).sum()
        return np.clip(a.cpu().numpy(), -1.0, 1.0), float(logp.item())

    def critic_value(self, obs):
        """obs: (n_cam, obs_dim) -> scalar value."""
        flat = torch.as_tensor(
            np.asarray(obs, dtype=np.float32).reshape(1, -1), device=self.device
        )
        with torch.no_grad():
            v = self.critic(flat)
        return float(v.item())

    # ------------------------------------------------------------------
    def collect_episode(self, env, max_steps, deterministic=False):
        """Run one episode, acting every frame_skip steps. Returns (returns, length)."""
        obs = env.reset()
        ep_ret = 0.0
        steps = 0
        done = False
        while not done and steps < max_steps:
            actions, logp = self.act_batch(obs, deterministic=deterministic)
            # one action spans frame_skip env steps (MATE example convention)
            for _ in range(self.frame_skip):
                if done or steps >= max_steps:
                    break
                obs_next, reward, done, info = env.step(actions)
                ep_ret += reward
                steps += 1
            obs = obs_next
        self.episode_count += 1
        return ep_ret, steps

    def collect_rollout(self, env, horizon, max_steps, deterministic=False):
        """Collect up to `horizon` env steps of transitions into the buffer
        (bounded by `max_steps` per episode). One call = one episode."""
        self._buffer.clear()
        obs = env.reset()
        steps = 0
        done = False
        while steps < horizon and steps < max_steps:
            value = self.critic_value(obs)
            actions, logp = self.act_batch(obs, deterministic=False)
            obs_next, reward, done, _ = env.step(actions)
            self._buffer.add(obs, actions, reward, value, done, logp)
            steps += 1
            self.step_count += 1
            obs = obs_next
            if done:
                break
        last_value = self.critic_value(obs) if not done else 0.0
        return self._buffer.finalize(last_value, done)

    # ------------------------------------------------------------------
    def update(self, buffer):
        obs, actions, rewards, adv, returns = buffer
        if len(returns) < 1:
            return 0.0, 0.0
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        adv = adv.to(self.device)
        returns = returns.to(self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        old_logp, _, _ = self.policy.evaluate_actions(obs, actions)
        old_logp = old_logp.detach()

        n = len(returns)
        total_pg = 0.0
        total_vf = 0.0
        for _ in range(self.epochs):
            idx = torch.randperm(n, device=self.device)
            for i in range(0, n, self.minibatch):
                ids = idx[i:i + self.minibatch]
                logp, entropy, _ = self.policy.evaluate_actions(obs[ids], actions[ids])
                log_ratio = (logp - old_logp[ids]).clamp(-20.0, 20.0)
                ratio = torch.exp(log_ratio)
                adv_b = adv[ids]
                pg1 = -ratio * adv_b
                pg2 = -torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv_b
                pg_loss = torch.max(pg1, pg2).mean()

                v_pred = self.critic(obs[ids].reshape(len(ids), -1))
                vf_loss = 0.5 * (v_pred - returns[ids]).pow(2).mean()
                entropy_loss = -entropy.mean()

                loss = pg_loss + self.value_coef * vf_loss + self.entropy_coef * entropy_loss
                self.opt.zero_grad()
                loss.backward()
                # Clip policy and critic gradients SEPARATELY. A single shared
                # clip_grad_norm over all params lets the (much larger) value
                # gradients scale the whole vector down, crushing the policy
                # gradient ~variance-ratio times and freezing the policy.
                params = list(self.policy.parameters()) + list(self.critic.parameters())
                for p in params:
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        p.grad.zero_()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.opt.step()
                total_pg += float(pg_loss.item())
                total_vf += float(vf_loss.item())
        return total_pg / max(self.epochs * max(1, n // self.minibatch), 1), \
            total_vf / max(self.epochs * max(1, n // self.minibatch), 1)

    # ------------------------------------------------------------------
    def save(self, path, tag=""):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "opt": self.opt.state_dict(),
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            "seed": self.seed,
            "tag": tag,
            "config": {
                "obs_dim": self.obs_dim, "act_dim": self.act_dim, "n_cam": self.n_cam,
            },
        }, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ck["policy"])
        self.critic.load_state_dict(ck["critic"])
        self.opt.load_state_dict(ck["opt"])
        self.step_count = ck.get("step_count", 0)
        self.episode_count = ck.get("episode_count", 0)
        return ck

    # ------------------------------------------------------------------
    def train(self, env, max_total_steps, max_steps_per_episode,
              horizon, eval_env=None, checkpoint_dir=None,
              checkpoint_interval=1000, eval_every=None, resume=None):
        """Train loop with periodic checkpoints and optional best-by-eval ckpt."""
        if not self.obs_normalizer.fitted:
            self.fit_obs_normalizer(env)
        if resume and os.path.exists(resume):
            self.load(resume)
            if self.logger:
                self.logger.info(f"[MAPPO] resumed from {resume} at step {self.step_count}")

        last_checkpoint = self.step_count
        best_eval = -np.inf
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        while self.step_count < max_total_steps:
            buffer = self.collect_rollout(env, horizon, max_steps_per_episode)
            pg, vf = self.update(buffer)
            if self.logger and (self.episode_count % self.log_every == 0):
                self.logger.info(
                    f"[MAPPO] step={self.step_count} ep={self.episode_count} "
                    f"pg={pg:.4f} vf={vf:.4f}"
                )

            if checkpoint_dir and (self.step_count - last_checkpoint) >= checkpoint_interval:
                self.save(os.path.join(checkpoint_dir, "latest.pt"), tag="latest")
                last_checkpoint = self.step_count
                if self.logger:
                    self.logger.info(f"[MAPPO] checkpoint at step {self.step_count}")

            if eval_env is not None and eval_every and \
                    (self.step_count % eval_every < checkpoint_interval):
                ret, _ = self.collect_episode(eval_env, max_steps_per_episode,
                                              deterministic=True)
                if ret > best_eval and checkpoint_dir:
                    best_eval = ret
                    self.save(os.path.join(checkpoint_dir, "best.pt"), tag="best")
                    if self.logger:
                        self.logger.info(f"[MAPPO] best eval return {ret:.3f}")

        if checkpoint_dir:
            self.save(os.path.join(checkpoint_dir, "latest.pt"), tag="final")
        return self.step_count
