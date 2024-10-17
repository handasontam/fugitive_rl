
import glob
import os
import time

from models.marl_ppo_mask import MARLMaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from common.subproc_vec_env import MARLSubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

import pettingzoo.utils
# from pettingzoo.classic import connect_four_v3
from gymnasium.spaces.utils import flatten_space, flatten, flatdim
from envs import fugitive
import numpy as np
import torch as th

def make_env(env_fn, seed=0, **env_kwargs):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = env_fn.env(**env_kwargs)
        env = SB3ActionMaskWrapper(env)
        env.reset(seed=seed)
        env = ActionMasker(env, mask_fn)
        return env
    set_random_seed(seed)
    return _init

class SB3ActionMaskWrapper(pettingzoo.utils.BaseWrapper):
    """Wrapper to allow PettingZoo environments to be used with SB3 illegal action masking."""

    def reset(self, seed=None, options=None):
        """Gymnasium-like reset function which assigns obs/action spaces to be the same for each agent.
        This is required as SB3 is designed for single-agent RL and doesn't expect obs/action spaces to be functions
        """
        super().reset(seed, options)

        self.observation_space = super().observation_space(self.possible_agents[0])
        self.action_space = super().action_space(self.possible_agents[0])

        # Return initial observation, info (PettingZoo AEC envs do not by default)
        return self.observe(self.agent_selection), {}

    def step(self, action):
        """Gymnasium-like step function, returning observation, reward, termination, truncation, info."""
        super().step(action)
        agent = self.agent_selection
        assert agent is not None
        observation = super().observe(agent)
        all_cumulative_rewards = [self._cumulative_rewards[self.agent_index_to_name(agent)] for agent in range(self.n_agents)]
        return observation, self._cumulative_rewards[agent], all_cumulative_rewards, self.terminations[agent], self.truncations[agent], self.infos[agent]
        # return super().last()

    def observe(self, agent):
        """Return only raw observation, removing action mask."""
        return super().observe(agent)
        # return flatten(super().observation_space(agent), super().observe(agent))

    def action_mask(self):
        """Separate function used in order to access the action mask."""
        # return self.infos[self.agent_selection]["action_mask"]
        return self.get_action_mask()


def mask_fn(env):
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.action_mask()


def train_action_mask(env_fn, steps=10_000, seed=0, num_envs=4, **env_kwargs):
    """Train a single model to play as each agent in a zero-sum game environment using invalid action masking."""
    env = env_fn.env(**env_kwargs)

    print(f"Starting training on {str(env.metadata['name'])}.")

    # Custom wrapper to convert PettingZoo envs to work with SB3 action masking
    env = MARLSubprocVecEnv([make_env(env_fn, seed + i, **env_kwargs) for i in range(num_envs)], start_method="fork")

    # env = SB3ActionMaskWrapper(env)

    # env.reset(seed=seed)  # Must call reset() in order to re-define the spaces

    # env = ActionMasker(env, mask_fn)  # Wrap to enable masking (SB3 function)
    # MARLMaskablePPO behaves the same as SB3's PPO unless the env is wrapped
    # with ActionMasker. If the wrapper is detected, the masks are automatically
    # retrieved and used when learning. Note that MARLMaskablePPO does not accept
    # a new action_mask_fn kwarg, as it did in an earlier draft.
    policy_kwargs = dict(net_arch={'pi': [128, 128, 128], 'vf': [128, 128]}, activation_fn=th.nn.ReLU, share_features_extractor=False)

    model = MARLMaskablePPO(MaskableMultiInputActorCriticPolicy, env, policy_kwargs=policy_kwargs, batch_size=4096, verbose=1, n_steps=4096 // num_envs, vf_coef=0.2)
    model.set_random_seed(seed)
    model.learn(total_timesteps=steps)

    # model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    model.save(f"{env_fn.env(**env_kwargs).metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")


    print("Model has been saved.")
    print(f"Finished training on {env_fn.env(**env_kwargs).metadata['name']}.\n")


    # print(f"Finished training on {str(env.unwrapped.metadata['name'])}.\n")

    env.close()


def eval_action_mask(env_fn, num_games=100, render_mode=None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs)
    # env = env_fn.env(**env_kwargs)

    print(
        f"Starting evaluation vs a random agent. Trained agent will play as {env.possible_agents[0]} and {env.possible_agents[1]} and {env.possible_agents[2]}. Random agent will play as {env.possible_agents[3]} and {env.possible_agents[4]}."
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = MARLMaskablePPO.load(latest_policy)

    agents = ["Fugitive", "Marshal"]
    scores = {agent: 0 for agent in agents}
    total_rewards = {agent: 0 for agent in agents}

    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            # Separate observation and action mask

            # observation = flatten(env.observation_space(env.agent_selection), obs)
            # action_mask = info["action_mask"]
            action_mask = np.array(env.get_action_mask(), dtype=np.int8)
            if termination or truncation:
                # If there is a winner, keep track, otherwise don't change the scores (tie)
                winner = "Fugitive" if env.rewards[env.possible_agents[0]] == 1 else "Marshal"
                scores[winner] += 1
                # Also track negative and positive rewards (penalizes illegal moves)
                total_rewards[winner] += 1
                break
            else:
                if obs['current_agent'] == 1:
                    act = env.action_space(agent).sample(action_mask)
                else:
                    # Note: PettingZoo expects integer actions # TODO: change chess to cast actions to type int?
                    act = int(
                        model.predict(
                            obs, action_masks=action_mask, deterministic=True
                        )[0]
                    )
            if render_mode == "human":
                env.render()
                print(act)
            env.step(act)
    env.close()

    # Avoid dividing by zero
    if sum(scores.values()) == 0:
        winrate = 0
    else:
        winrate = scores["Fugitive"] / sum(scores.values())
    print("Total rewards (incl. negative rewards): ", total_rewards)
    print("Winrate as Fugitive vs Random Marshal: ", winrate)
    print("Final scores: ", scores)
    return total_rewards, winrate, scores


if __name__ == "__main__":
    env_fn = fugitive


    # Evaluation/training hyperparameter notes:
    # 10k steps: Winrate:  0.76, loss order of 1e-03
    # 20k steps: Winrate:  0.86, loss order of 1e-04
    # 40k steps: Winrate:  0.86, loss order of 7e-06

    # Train a model against itself (takes ~20 seconds on a laptop CPU)
    # train_action_mask(env_fn, steps=200_480, num_envs=1, seed=0, **env_kwargs)

    env_kwargs = {}

    train_action_mask(env_fn, steps=2_000_480, num_envs=64, seed=0, **env_kwargs)

    # Evaluate 100 games against a random agent (winrate should be ~80%)
    eval_action_mask(env_fn, num_games=1000, render_mode=None, **env_kwargs)

    # Watch two games vs a random agent
    eval_action_mask(env_fn, num_games=1, render_mode="human", **env_kwargs)