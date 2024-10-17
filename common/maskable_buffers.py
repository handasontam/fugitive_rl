from typing import Generator, NamedTuple, Optional, Union, Dict

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.vec_env import VecNormalize
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer, MaskableRolloutBuffer



class MARLMaskableDictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    action_masks: th.Tensor
    agent_ids: th.Tensor

class MARLMaskableDictRolloutBuffer(MaskableDictRolloutBuffer):
    """
    Dict Rollout buffer used in on-policy algorithms like A2C/PPO.
    Extends the RolloutBuffer to use dictionary observations

    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param n_envs: Number of parallel environments
    """

    action_masks: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        n_agents: int, 
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        n_envs: int = 1,
    ):
        self.n_agents = n_agents
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, n_envs=n_envs)

    def reset(self) -> None:
        if isinstance(self.action_space, spaces.Discrete):
            mask_dims = self.action_space.n
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            mask_dims = sum(self.action_space.nvec)
        elif isinstance(self.action_space, spaces.MultiBinary):
            assert isinstance(self.action_space.n, int), (
                f"Multi-dimensional MultiBinary({self.action_space.n}) action space is not supported. "
                "You can flatten it instead."
            )
            mask_dims = 2 * self.action_space.n  # One mask per binary outcome
        else:
            raise ValueError(f"Unsupported action space {type(self.action_space)}")

        self.mask_dims = mask_dims
        self.action_masks = np.ones((self.buffer_size, self.n_envs, self.mask_dims), dtype=np.float32)
        self.agent_ids = np.zeros((self.buffer_size, self.n_envs), dtype=np.int8)

        super().reset()

    def add(self, *args, action_masks: Optional[np.ndarray] = None, agent_id: Optional[int] = None, **kwargs) -> None:
        """
        :param action_masks: Masks applied to constrain the choice of possible actions.
        """
        if action_masks is not None:
            self.action_masks[self.pos] = action_masks.reshape((self.n_envs, self.mask_dims))
        if agent_id is not None:
            self.agent_ids[self.pos] = agent_id

        super().add(*args, **kwargs)
    
    def back_assign_agent_rewards(self, env_id: int, all_rewards: np.ndarray) -> None:
        """
        Back-assign rewards to the rewards buffer.
        This is used to back-assign rewards when the episode is over.
        rewards: Dict[str, float], the rewards for each agent in the environment
        """
        back_assigned = [False] * self.n_agents
        pos = self.pos-1
        while not all(back_assigned) and pos >= 0:
            agent_id = self.agent_ids[pos, env_id]  
            if not back_assigned[agent_id]:
                self.rewards[pos, env_id] = all_rewards[agent_id]
                back_assigned[agent_id] = True
            pos = pos - 1

    def get(self, batch_size: Optional[int] = None) -> Generator[MARLMaskableDictRolloutBufferSamples, None, None]:  # type: ignore[override]
        assert self.full, ""
        indices = np.random.permutation(self.trimmed_buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            _tensor_names = ["actions", "values", "log_probs", "advantages", "returns", "action_masks", "agent_ids"]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.trimmed_buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.trimmed_buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size
    
    def get_win_rate(self):
        win_rate = np.zeros(self.n_agents, dtype=np.float32)
        for agent_id in range(self.n_agents):
            agent_mask = self.agent_ids == agent_id
            agent_rewards = self.rewards[agent_mask]
            win_count = np.sum(agent_rewards == 1)
            lose_count = np.sum(agent_rewards == -1)
            win_rate[agent_id] = win_count / (win_count + lose_count)
        return win_rate

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> MARLMaskableDictRolloutBufferSamples:  # type: ignore[override]
        return MARLMaskableDictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations.items()},
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
            action_masks=self.to_torch(self.action_masks[batch_inds].reshape(-1, self.mask_dims)),
            agent_ids=self.to_torch(self.agent_ids[batch_inds].reshape(-1, 1)),
        )

#     def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
#         """
#         Post-processing step: compute the lambda-return (TD(lambda) estimate)
#         and GAE(lambda) advantage.

#         Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
#         to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
#         where R is the sum of discounted reward with value bootstrap
#         (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

#         The TD(lambda) estimator has also two special cases:
#         - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
#         - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

#         For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

#         :param last_values: state value estimation for the last step (one for each env)
#         :param dones: if the last step was a terminal step (one bool for each env).
#         """
#         # Convert to numpy
#         last_values = last_values.clone().cpu().numpy().flatten()

#         last_gae_lam = 0
#         for step in reversed(range(self.buffer_size)):
#             if step == self.buffer_size - 1:
#                 next_non_terminal = 1.0 - dones
#                 next_values = last_values
#             else:
#                 next_non_terminal = 1.0 - self.episode_starts[step + 1]
#                 next_values = self.values[step + 1]
#             delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
#             last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
#             self.advantages[step] = last_gae_lam
#         # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
#         # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
#         self.returns = self.advantages + self.values

    def compute_returns_and_advantage(self) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage for multi-agent settings.
        """

        for i in range(self.n_envs):
            agent_ids = self.agent_ids[:, i]
            for agent_id in range(self.n_agents):
                agent_ids_mask = agent_ids == agent_id
                values = self.values[agent_ids_mask,i]
                episode_starts = self.episode_starts[agent_ids_mask,i]
                rewards = self.rewards[agent_ids_mask,i]
                last_value = self.values[agent_ids_mask,i][-1]
                done = self.episode_starts[agent_ids_mask,i][-1]
                advantages, returns = self.compute_returns_and_advantage_for_agent(values, episode_starts, rewards, last_value, done)
                self.advantages[agent_ids_mask,i] = advantages
                self.returns[agent_ids_mask,i] = returns


        # Compute last_indices for trimming the buffer
        agent_ids = self.agent_ids.reshape(-1, self.n_envs)
        last_indices = np.zeros((self.n_envs, self.n_agents), dtype=np.int32)
        
        for agent in range(self.n_agents):
            agent_mask = agent_ids == agent
            last_indices[:, agent] = np.where(agent_mask.any(axis=0), agent_mask.shape[0] - 1 - np.flip(agent_mask, axis=0).argmax(axis=0), -1)
        trimmed_buffer_size = last_indices.min()

        for key, obs in self.observations.items():
            self.observations[key] = self.observations[key][:trimmed_buffer_size]
        
        _tensor_names = ["actions", "values", "log_probs", "advantages", "returns", "action_masks", "agent_ids"]
        for tensor in _tensor_names:
            self.__dict__[tensor] = self.__dict__[tensor][:trimmed_buffer_size]
        
        self.trimmed_buffer_size = trimmed_buffer_size


    def compute_returns_and_advantage_for_agent(self, values: np.ndarray, episode_starts: np.ndarray, rewards: np.ndarray, last_value: float, done: int) -> None:
        advantages = np.zeros_like(values)
        returns = np.zeros_like(values)

        n_steps = len(values)

        last_gae_lam = 0
        for step in reversed(range(n_steps)):
            if step == n_steps - 1:
                next_non_terminal = 1.0 - done
                next_values = last_value
            else:
                next_non_terminal = 1.0 - episode_starts[step + 1]
                next_values = values[step + 1]
            delta = rewards[step] + next_values * next_non_terminal - values[step]
            last_gae_lam = delta + self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam
        returns = advantages + values
        return advantages, returns
        