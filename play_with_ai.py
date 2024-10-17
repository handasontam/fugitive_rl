import glob
import time
from models.marl_ppo_mask import MARLMaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from envs import fugitive
from gymnasium.spaces.utils import flatten
import numpy as np
from colorama import Fore, Back, Style, init
import os
import torch as th

init(autoreset=True)

def get_user_action(env, action_mask):
    valid_actions = np.where(action_mask)[0]
    while True:
        if env.agent_selection == fugitive.PlayerRole.FUGITIVE_DRAW:
            print(f"\n{Fore.YELLOW}Available actions: {Fore.GREEN}{valid_actions}")
        elif env.agent_selection == fugitive.PlayerRole.FUGITIVE_HIHDEOUT:
            print(f"\n{Fore.YELLOW}Available actions: {Fore.GREEN}{valid_actions-8}")
        else:
            print(f"\n{Fore.YELLOW}Available actions: {Fore.GREEN}{valid_actions-52}")
        action = int(input(f"{Fore.CYAN}Enter your action: "))

        if env.agent_selection == fugitive.PlayerRole.FUGITIVE_HIHDEOUT:
            action += 8
        elif env.agent_selection == fugitive.PlayerRole.FUGITIVE_SPRINT:
            action += 52
        try:
            action = int(action)
            if action in valid_actions:
                return action
            else:
                print(f"{Fore.RED}Invalid action. Please choose from the available actions.")
        except ValueError:
            print(f"{Fore.RED}Invalid input. Please enter a number.")

def print_policy_distribution(model, observation, action_mask):
    action_masks = np.expand_dims(action_mask, axis=0)
    obs = {k: np.expand_dims(v, axis=0) for k, v in observation.items()}
    dist = model.policy.get_distribution(obs)
    probs = dist.distribution.probs[0].detach().numpy()
    masked_probs = np.where(action_mask, probs, 0)
    masked_probs /= masked_probs.sum()
    
    print(f"\n{Fore.MAGENTA}AI Policy Distribution:")
    for action, prob in enumerate(masked_probs):
        if action_mask[action]:
            print(f"{Fore.CYAN}Action {action}: {Fore.GREEN}{prob:.4f}")

def play_with_ai():
    env = fugitive.env(render_mode='human')
    env.reset()

    try:
        latest_policy = max(glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime)
        model = MARLMaskablePPO.load(latest_policy)
        print(f"{Fore.GREEN}Loaded policy: {latest_policy}")
    except ValueError:
        print(f"{Fore.RED}Policy not found. Please train a model first.")
        return

    observation, info = env.reset()

    while True:
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                if env.rewards[env.possible_agents[0]] == 1:
                    print(f"\n{Fore.GREEN}Game Over! Fugitive (You) won!")
                else:
                    print(f"\n{Fore.RED}Game Over! Marshal (AI) won!")
                return

            action_mask = np.array(env.get_action_mask(), dtype=np.int8)

            if agent in [fugitive.PlayerRole.FUGITIVE_DRAW, fugitive.PlayerRole.FUGITIVE_HIHDEOUT, fugitive.PlayerRole.FUGITIVE_SPRINT]:
                env.render()
                action = get_user_action(env, action_mask)
            else:
                # print_policy_distribution(model, observation, action_mask)
                action = int(model.predict(observation, action_masks=action_mask, deterministic=False)[0])

            env.step(action)

        #     time.sleep(1)  # Add a small delay for better readability

if __name__ == "__main__":
    play_with_ai()