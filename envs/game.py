import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Sequence, Box, Discrete, MultiBinary, MultiDiscrete
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from gymnasium.wrappers import FlattenObservation
from gymnasium.spaces.utils import flatten_space, flatten, flatdim
from functools import lru_cache
from enum import StrEnum
from typing import Optional, List, Set
import networkx as nx
from colorama import Fore, Back, Style, init
init(autoreset=True)

class PlayerRole(StrEnum):
    FUGITIVE_DRAW = 'fugitive_0'
    FUGITIVE_HIHDEOUT = 'fugitive_1'
    FUGITIVE_SPRINT = 'fugitive_2'
    MARSHAL_DRAW = 'marshal_0'
    MARSHAL_GUESS = 'marshal_1'

MAX_HAND_SIZE = 26 # fugitive get 9 cards at start, each player can draw at most (42-9)/2=17 cards in total, so the maximum hand size is 17+9=26
PASS_HIDEOUT_ACTION = 43  # passing hideout
LAST_HIDEOUT_ACTION = 42  # playing the last hideout
PASS_DRAW_ACTION = 3  # passing draw because there is no card in deck
STOP_SPRINT_ACTION = 0
HIDDEN_HIDEOUT = 43

def env():
    env = Fugitive()
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    env = FlattenObservation(env)
    return env


def multibinary_to_cards(multibinary):
    return np.where(multibinary==1)[0]



class MarshalNotes:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.graph.add_node((0, 0))  # (depth, value)
        self.hideouts_range = np.zeros((MAX_HAND_SIZE, 43), dtype=np.int8)
        self.hideouts_range[0, 0] = 1
        self.n_hideouts = 1

    def add_new_hideout(self, n_sprint: int, filtered_vals: List[int]) -> None:
        prev_depth = self.n_hideouts - 1
        new_depth = self.n_hideouts
        
        nodes_from_prev_depth = [node for node in self.graph.nodes() if node[0] == prev_depth]
        for prev_node in nodes_from_prev_depth:
            prev_value = prev_node[1]
            min_new_value = prev_value + 1
            max_new_value = min(prev_value + 3 + 2 * n_sprint + 1, 42)
            
            for new_value in range(min_new_value, max_new_value):
                if new_value in filtered_vals:  # Eliminated seen values
                    continue
                new_node = (new_depth, new_value)
                self.graph.add_edge(prev_node, new_node)
                self.hideouts_range[new_depth, new_value] = 1
        self.n_hideouts += 1
    
    def add_escape_hideout(self, n_sprint: int) -> None:
        prev_depth = self.n_hideouts - 1
        new_depth = self.n_hideouts

        nodes_from_prev_depth = [node for node in self.graph.nodes() if node[0] == prev_depth]
        can_reach_42 = [node for node in nodes_from_prev_depth if node[1] + 2 * n_sprint + 3 >= 42]
        for node in can_reach_42:
            self.graph.add_edge(node, (new_depth, 42))
        self.hideouts_range[new_depth, 42] = 1
        self.n_hideouts += 1
        self._remove_unreachable_nodes()

    def eliminate_vals(self, vals: List[int]) -> None:
        # Remove nodes with the revealed value
        nodes_to_remove = [(d, v) for d, v in self.graph.nodes() if v in vals and d > 0]
        self.graph.remove_nodes_from(nodes_to_remove)
        self.hideouts_range[:, vals] = 0
        self._remove_unreachable_nodes()

    def hideout_revealed(self, depth: int, hideout: int) -> None:
        # Remove all nodes at the given depth except the revealed hideout
        nodes_to_remove = [(d, v) for d, v in self.graph.nodes() if d == depth and v != hideout]
        self.graph.remove_nodes_from(nodes_to_remove)
        
        # Update hideouts_range for this depth
        self.hideouts_range[depth, :] = 0
        self.hideouts_range[depth, hideout] = 1
        
        # Remove unreachable nodes
        self._remove_unreachable_nodes()
    
    def _remove_unreachable_nodes(self):
        # Remove nodes unreacheable from root
        for depth in range(1, self.n_hideouts):
            nodes_at_depth = [node for node in self.graph.nodes() if node[0] == depth]
            for node in nodes_at_depth:
                if self.graph.in_degree(node) == 0:
                    self.graph.remove_node(node)
                    self.hideouts_range[node[0], node[1]] = 0
        
        # Remove nodes unreachable to the leaf at depth n_hideouts
        for depth in range(self.n_hideouts-2, 0, -1):
            nodes_at_depth = [node for node in self.graph.nodes() if node[0] == depth]
            for node in nodes_at_depth:
                if self.graph.out_degree(node) == 0:
                    self.graph.remove_node(node)
                    self.hideouts_range[node[0], node[1]] = 0


    def get_hideouts_range(self) -> np.ndarray:
        return self.hideouts_range
    
    def get_hideouts_range_for_agent(self) -> np.ndarray:
        return np.any(self.hideouts_range[:self.n_hideouts], axis=0)



class Fugitive(AECEnv):
    metadata = {"render_modes": ["human"], "name": "fugitive_v0"}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.possible_agents = [PlayerRole.FUGITIVE_DRAW, PlayerRole.FUGITIVE_HIHDEOUT, PlayerRole.FUGITIVE_SPRINT, PlayerRole.MARSHAL_DRAW, PlayerRole.MARSHAL_GUESS]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))

        # Define action spaces
        self.draw_action_space = Discrete(4)  # 0, 1, 2 for selecting deck, 3 means pass (only allowed when no cards left in all decks)
        self.fugitive_hideout_action_space = Discrete(44)  # 0-42 for placing hideouts, 43 for pass
        self.fugitive_sprint_action_space = Discrete(42)  # 1-41 for sprinting, 0 means not sprinting
        self.marshal_guess_action_space = Discrete(42)  # 0-41 for guessing hideouts

    @lru_cache(maxsize=None)
    def observation_space(self, agent):
        if agent in [PlayerRole.FUGITIVE_DRAW, PlayerRole.FUGITIVE_HIHDEOUT, PlayerRole.FUGITIVE_SPRINT]:
            return spaces.Dict({
                'fugitive_center_row': MultiDiscrete([43] * MAX_HAND_SIZE), # can only be 1-42, 0 is for padding
                'fugitive_center_row_sprint': MultiBinary((MAX_HAND_SIZE, 42)), # [i][j] = 1 means j is sprinted at the i-th card in the center row, can only be 1-41, 0 is dummy
                'marshal_center_row': MultiDiscrete([44] * MAX_HAND_SIZE),  # can only be 1-42, 43 means hidden, 0 is for unplaced hideout
                'marshal_center_row_sprint_size': MultiDiscrete([MAX_HAND_SIZE] * MAX_HAND_SIZE),  # number of sprint used for each card in the center row
                'fugitive_hand': MultiBinary(43), # can only be 1-42, 0 is dummy
                'marshal_hand_size': Discrete(MAX_HAND_SIZE),
                'fugitive_hand_size': Discrete(MAX_HAND_SIZE),
                'last_hideout': Discrete(43),
                'deck_low_size': Discrete(15), # can only be 0-14
                'deck_medium_size': Discrete(15),
                'deck_high_size': Discrete(14),  # can only be 0-13
                'n_hideouts': Discrete(MAX_HAND_SIZE),
                'guessed': MultiBinary(43), # Indicating if fugitive has guessed any hideout, is equivalent to np.any(guessed_center_row, axis=1)
                'guessed_center_row': MultiBinary((MAX_HAND_SIZE, 42)),  # [i][j] = 1 means j is guessed at the i-th card in the center row
                'revealed_hideout': MultiBinary(43),  # revealed_hideout[i] = 1 if card-i in center row is revealed as a hideout
                'revealed_sprint': MultiBinary(42),  # revealed_sprint[i] = 1 if card-i in center row is revealed by a sprint, card 42 cannot be used as sprint
                'is_30_or_above_revealed': Discrete(2),  # 0 if no hideout is revealed above 30, 1 if at least one hideout above 30 is revealed
                'is_manhunt': Discrete(2),  # 0 if not in manhunt, 1 if in manhunt
                'observation': Discrete(1)
            })
        else:  # marshal
            return spaces.Dict({
                'marshal_center_row': MultiDiscrete([44] * MAX_HAND_SIZE),  # can only be 1-42, 43 means hidden, 0 is for unplaced hideout
                'marshal_center_row_sprint_size': MultiDiscrete([MAX_HAND_SIZE] * MAX_HAND_SIZE),  # number of sprint used for each card in the center row
                'marshal_hand': MultiBinary(42),  # can only be 4-41, 0 - 3 are dummies
                'marshal_hand_size': Discrete(MAX_HAND_SIZE),
                'fugitive_hand_size': Discrete(MAX_HAND_SIZE),
                'deck_low_size': Discrete(15), # can only be 0-14
                'deck_medium_size': Discrete(15),
                'deck_high_size': Discrete(14),  # can only be 0-13
                'n_hideouts': Discrete(MAX_HAND_SIZE),
                'guessed': MultiBinary(43), 
                'guessed_center_row': MultiBinary((MAX_HAND_SIZE, 42)), 
                'revealed_hideout': MultiBinary(43),
                'revealed_sprint': MultiBinary(42), 
                'hideouts_range': MultiBinary((MAX_HAND_SIZE, 43)),  # [i][j] = 1 means the i-th card can potentially be j
                'is_30_or_above_revealed': Discrete(2),  # 0 if no hideout is revealed above 30, 1 if at least one hideout above 30 is revealed
                'is_manhunt': Discrete(2),  # 0 if not in manhunt, 1 if in manhunt
                'observation': Discrete(1)
            })

    def _get_obs(self):
        fugitive_obs = {
            'fugitive_center_row': self.fugitive_center_row, 
            'fugitive_center_row_sprint': self.fugitive_center_row_sprint,
            'marshal_center_row': self.marshal_center_row,
            'marshal_center_row_sprint_size': self.marshal_center_row_sprint_size,
            'fugitive_hand': self.fugitive_hand, 
            'marshal_hand_size': np.sum(self.marshal_hand), 
            'fugitive_hand_size': np.sum(self.fugitive_hand),
            'guessed': self.guessed, 
            'guessed_center_row': self.guessed_center_row,
            'revealed_hideout': self.revealed_hideout,
            'revealed_sprint': self.revealed_sprint,
            'last_hideout': self.last_hideout, 
            'deck_low_size': len(self.deck_low), 
            'deck_medium_size': len(self.deck_medium),
            'deck_high_size': len(self.deck_high),
            'n_hideouts': self.n_hideouts,
            'is_30_or_above_revealed': self._is_30_or_above_revealed(),
            'is_manhunt': self.is_manhunt,
            'observation': np.int64(0)  # mainly for passing the api_test, int64 to match the observation space, 
        }
        
        marshal_obs = {
            'marshal_center_row': self.marshal_center_row,
            'marshal_center_row_sprint_size': self.marshal_center_row_sprint_size,
            'marshal_hand': self.marshal_hand, 
            'marshal_hand_size': np.sum(self.marshal_hand), 
            'fugitive_hand_size': np.sum(self.fugitive_hand),
            'guessed': self.guessed, 
            'guessed_center_row': self.guessed_center_row,
            'revealed_hideout': self.revealed_hideout,
            'revealed_sprint': self.revealed_sprint,
            'deck_low_size': len(self.deck_low),
            'deck_medium_size': len(self.deck_medium),
            'deck_high_size': len(self.deck_high),
            'n_hideouts': self.n_hideouts,
            'is_30_or_above_revealed': self._is_30_or_above_revealed(),
            'hideouts_range': self.marshal_notes.get_hideouts_range(),
            'is_manhunt': self.is_manhunt,
            'observation': np.int64(0)
        }

        return {
            PlayerRole.FUGITIVE_DRAW: fugitive_obs,
            PlayerRole.FUGITIVE_HIHDEOUT: fugitive_obs,
            PlayerRole.FUGITIVE_SPRINT: fugitive_obs,
            PlayerRole.MARSHAL_DRAW: marshal_obs,
            PlayerRole.MARSHAL_GUESS: marshal_obs
        }
        

    def action_space(self, agent):
        if agent == PlayerRole.FUGITIVE_DRAW or agent == PlayerRole.MARSHAL_DRAW:  # Draw phase
            return self.draw_action_space
        if agent == PlayerRole.FUGITIVE_HIHDEOUT:
            return self.fugitive_hideout_action_space
        if agent == PlayerRole.FUGITIVE_SPRINT:
            return self.fugitive_sprint_action_space
        if agent == PlayerRole.MARSHAL_GUESS:
            return self.marshal_guess_action_space

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.fugitive_center_row = np.zeros(MAX_HAND_SIZE, dtype=np.int8)  # List for fugitive's observation
        self.fugitive_center_row_sprint = np.zeros((MAX_HAND_SIZE, 42), dtype=np.int8)
        self.marshal_center_row = np.zeros(MAX_HAND_SIZE, dtype=np.int8)  # List for marshal's observation
        self.marshal_center_row_sprint_size = np.zeros(MAX_HAND_SIZE, dtype=np.int8)
        self.revealed_hideout = np.zeros(43, dtype=np.int8)
        self.revealed_hideout[0] = 1  # 0 is revealed at the beginning
        self.revealed_sprint = np.zeros(42, dtype=np.int8)
        self.n_hideouts= 1
        self.hideout2center_row_index = dict()
        self.hideout2center_row_index[0] = 0  # 0 is revealed at the beginning

        self.deck_low = list(range(4, 15))
        self.deck_medium = list(range(15, 29))
        self.deck_high = list(range(29, 42))
        np.random.shuffle(self.deck_low)
        np.random.shuffle(self.deck_medium)
        np.random.shuffle(self.deck_high)
        
        fugitive_opening_hand = [1, 2, 3, 42] + self.deck_low[:3] + self.deck_medium[:2]
        self.fugitive_hand = np.zeros(43, dtype=np.int8)
        self.fugitive_hand[fugitive_opening_hand] = 1

        self.marshal_hand = np.zeros(42, dtype=np.int8)

        self.guessed = np.zeros(43, dtype=np.int8)
        self.guessed_center_row = np.zeros((MAX_HAND_SIZE, 42), dtype=np.int8)
        
        self.deck_low = self.deck_low[3:]
        self.deck_medium = self.deck_medium[2:]
        
        self.last_hideout = 0

        # Initialize hideouts_range 
        self.marshal_notes = MarshalNotes()

        self.is_manhunt = False

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.observation = self._get_obs()
        self.action_masks = self._get_action_mask()
        self.infos[self.agent_selection]["action_masks"] = np.array(self.action_masks, dtype=np.int8)

        return self.observation, self.infos

    def step(self, action: int) -> None:
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            return self._was_dead_step(action)

        # Apply the action
        self._take_action(action)
        
        # Update rewards
        self._accumulate_rewards()
        
        # Update terminations and truncations
        if self._check_termination():
            self.terminations = {agent: True for agent in self.agents}
            return
        
        # Update observations
        self.observation = self._get_obs()

        # Update agent selector
        if not self.is_manhunt:
            if self.agent_selection == PlayerRole.FUGITIVE_SPRINT and action != STOP_SPRINT_ACTION:
                    pass
            elif self.agent_selection == PlayerRole.FUGITIVE_HIHDEOUT and action == PASS_HIDEOUT_ACTION:
                # Skip the fugitive's sprint turn to marshal's draw turn
                self._agent_selector.next()
                self.agent_selection = self._agent_selector.next()
            else:
                self.agent_selection = self._agent_selector.next()
        else:  # Manhunt
            if self.agent_selection == PlayerRole.FUGITIVE_HIHDEOUT:
                self.agent_selection = self._agent_selector.next()
            elif self.agent_selection == PlayerRole.FUGITIVE_SPRINT:
                if action == STOP_SPRINT_ACTION:
                    # Skip the marshal's draw turn to marshal's guess turn (manhunt)
                    self._agent_selector.next()
                    self.agent_selection = self._agent_selector.next()
                else:
                    pass
            elif self.agent_selection == PlayerRole.MARSHAL_GUESS:
                pass  # Stay on MARSHAL_GUESS during manhunt
            else:
                raise ValueError(f"In Manhunt, stage {self.agent_selection} is not allowed")
    
        self.action_masks = self._get_action_mask()
        if sum(self.action_masks) == 1:  # forced move
            self.step(self.action_masks.index(1))

        self.infos = {agent: {} for agent in self.agents}
        self.infos[self.agent_selection]["action_masks"] = np.array(self.action_masks, dtype=np.int8)

    def _take_action(self, action: int) -> None:
        # The action must be legal
        # Draw phase
        if self.agent_selection == PlayerRole.FUGITIVE_DRAW or self.agent_selection == PlayerRole.MARSHAL_DRAW:
            if action == PASS_DRAW_ACTION:  # Pass
                return
            selected_deck = [self.deck_low, self.deck_medium, self.deck_high][action]
            drawn_card = selected_deck.pop()
            if self.agent_selection == PlayerRole.FUGITIVE_DRAW:
                self.fugitive_hand[drawn_card] = 1
            else:  # Marshal
                self.marshal_hand[drawn_card] = 1
                self.marshal_notes.eliminate_vals([drawn_card])
            return

        if self.agent_selection == PlayerRole.FUGITIVE_HIHDEOUT:
            if action == PASS_HIDEOUT_ACTION:  # Pass
                return
            self.fugitive_center_row[self.n_hideouts] = action
            self.hideout2center_row_index[action] = self.n_hideouts
            self.marshal_center_row[self.n_hideouts] = HIDDEN_HIDEOUT if action != LAST_HIDEOUT_ACTION else action  # Hidden for marshal, unless it is the last hideout
            self.fugitive_hand[action] = 0  # remove card from fugitive hand
            self.last_hideout = action
            if action == LAST_HIDEOUT_ACTION and not self._is_30_or_above_revealed():
                self.is_manhunt = True
        
        if self.agent_selection == PlayerRole.FUGITIVE_SPRINT:
            if action == STOP_SPRINT_ACTION:
                # Update hideouts_range for the hideout
                n_sprint = self.marshal_center_row_sprint_size[self.n_hideouts]
                hideout = self.fugitive_center_row[self.n_hideouts]
                if hideout == LAST_HIDEOUT_ACTION:
                    self.revealed_hideout[hideout] = 1
                    self.marshal_notes.add_escape_hideout(n_sprint=n_sprint)
                else:
                    revealed_cards = multibinary_to_cards(self.revealed_hideout[:LAST_HIDEOUT_ACTION].astype(bool) | self.revealed_sprint.astype(bool) | self.marshal_hand.astype(bool))
                    self.marshal_notes.add_new_hideout(n_sprint=n_sprint, filtered_vals=revealed_cards)
                self.n_hideouts+= 1
                return
            self.fugitive_center_row_sprint[self.n_hideouts, action] = 1
            self.marshal_center_row_sprint_size[self.n_hideouts] += 1
            self.fugitive_hand[action] = 0

        if self.agent_selection == PlayerRole.MARSHAL_GUESS:
            self.guessed[action] = 1
            self.guessed_center_row[:self.n_hideouts, action] = 1
            if action in self.hideout2center_row_index:  # Correct guess
                # Reveal the hideout
                revealed_index = self.hideout2center_row_index[action]
                # Update hideouts_range 
                self.marshal_notes.hideout_revealed(depth=revealed_index, hideout=action)
                # Reveal hideouts with only one possible range
                one_possible_range_indices = multibinary_to_cards(np.sum(self.marshal_notes.get_hideouts_range(), axis=1) == 1)
                self._reveal_one_possible_hideouts(one_possible_range_indices)
            else:
                self.marshal_notes.eliminate_vals([action])
                if self.is_manhunt:
                    self.is_manhunt = False  # End manhunt if guess is incorrect
    
    def _reveal_one_possible_hideouts(self, one_possible_range_indices: np.ndarray) -> None:
        # Reveal all hideouts in one_possible_range_indices
        revealed_hideouts = self.fugitive_center_row[one_possible_range_indices]
        self.marshal_center_row[one_possible_range_indices] = revealed_hideouts
        self.revealed_hideout[revealed_hideouts] = 1
        revealed_sprints = np.sum(self.fugitive_center_row_sprint[one_possible_range_indices], axis=0)  # all sprints revealed in all hideouts
        self.revealed_sprint[revealed_sprints.astype(bool)] = 1
        self.marshal_notes.eliminate_vals(multibinary_to_cards(revealed_sprints))  # Eliminate all sprints revealed in this hideout


    def _is_valid_hideout(self, hideout: int) -> bool:
        if not self.fugitive_hand[hideout]:
            return False

        # Check if hideout is in range
        lower_bound = self.last_hideout + 1
        total_sprint = self._get_total_sprint(multibinary_to_cards(self.fugitive_hand[:42]))  # card 42 cannot be used as sprint
        if hideout == LAST_HIDEOUT_ACTION:
            upper_bound = self.last_hideout + 3 + total_sprint
        else:
            hide_out_sprint = 2 - hideout % 2
            upper_bound = self.last_hideout + 3 + total_sprint - hide_out_sprint
        return lower_bound <= hideout <= upper_bound
    
    def _get_total_sprint(self, cards: np.ndarray) -> int:
        return np.sum(2 - cards % 2)

    def _is_30_or_above_revealed(self) -> bool:
        return ((30 <= self.marshal_center_row) & (self.marshal_center_row <= 41)).any()

    def _accumulate_rewards(self):
        if self._check_termination():
            if LAST_HIDEOUT_ACTION in self.hideout2center_row_index and not self.is_manhunt:  # Fugitive escaped
                print("Fugitive escaped!")
                self.rewards = {
                    PlayerRole.FUGITIVE_DRAW: 1, 
                    PlayerRole.FUGITIVE_HIHDEOUT: 1, 
                    PlayerRole.FUGITIVE_SPRINT: 1,  
                    PlayerRole.MARSHAL_DRAW: -1, 
                    PlayerRole.MARSHAL_GUESS: -1
                }
            else:  # Marshal caught the fugitive
                print("Marshal caught the fugitive!")
                self.rewards = {
                    PlayerRole.FUGITIVE_DRAW: -1, 
                    PlayerRole.FUGITIVE_HIHDEOUT: -1, 
                    PlayerRole.FUGITIVE_SPRINT: -1,  
                    PlayerRole.MARSHAL_DRAW: 1, 
                    PlayerRole.MARSHAL_GUESS: 1
                }
        for agent in self.possible_agents:
            self._cumulative_rewards[agent] += self.rewards[agent]

    def _check_termination(self):
        # Game ends if fugitive reaches 42 and manhunt is over, or all hideouts are revealed
        return (LAST_HIDEOUT_ACTION in self.hideout2center_row_index and not self.is_manhunt) or (self.last_hideout > 0 and HIDDEN_HIDEOUT not in self.marshal_center_row)


    def observe(self, agent):
        return self._get_obs()[agent]
    
    def _get_action_mask(self):
        match self.agent_selection:
            case PlayerRole.FUGITIVE_DRAW:
                is_cards_left = [len(deck) > 0 for deck in [self.deck_low, self.deck_medium, self.deck_high]]
                if any(is_cards_left):
                    return is_cards_left + [False]  # not allow pass
                else:
                    return is_cards_left + [True]  # can only pass
            case PlayerRole.FUGITIVE_HIHDEOUT:
                # First turn not allow pass
                if self.n_hideouts == 1:
                    return [self.fugitive_hand[card] and self._is_valid_hideout(card) for card in range(43)] + [False]
                else:
                    return [self.fugitive_hand[card] and self._is_valid_hideout(card) for card in range(43)] + [True]  # Last True is for pass action
            case PlayerRole.FUGITIVE_SPRINT:
                return [self._can_stop_sprint()] + list(self.fugitive_hand[1:42])
            case PlayerRole.MARSHAL_DRAW:
                is_cards_left = [len(deck) > 0 for deck in [self.deck_low, self.deck_medium, self.deck_high]]
                if any(is_cards_left):
                    return is_cards_left + [False]  # not allow pass
                else:
                    return is_cards_left + [True]  # can only pass
            case PlayerRole.MARSHAL_GUESS:
                # Guess if in any possible range
                numbers_in_range = self.marshal_notes.get_hideouts_range_for_agent()
                not_revealed_mask = ~self.revealed_hideout.astype(bool)
                legal_mask = (numbers_in_range & not_revealed_mask)[:LAST_HIDEOUT_ACTION]  # should not guess 42
                return list(legal_mask)

    def _can_stop_sprint(self):
        cur = self.n_hideouts
        target_hideout = self.fugitive_center_row[cur]
        previous_hideout = self.fugitive_center_row[cur-1]
        cur_sprint_cards = multibinary_to_cards(self.fugitive_center_row_sprint[cur,:])
        current_sprint = self._get_total_sprint(cur_sprint_cards)
        return previous_hideout + 3 + current_sprint >= target_hideout

    def render(self):
        if self.render_mode == "human":

            print(f'\n{Fore.CYAN}{"="*60}')
            print(f"{Fore.YELLOW}Current Player: {Fore.GREEN}{'🚔 Marshal' if self.agent_selection in [PlayerRole.MARSHAL_DRAW, PlayerRole.MARSHAL_GUESS] else '🦹‍ Fugitive'}")
            print(f"{Fore.YELLOW}Current Phase: {Fore.GREEN}{self.agent_selection}")
            print(f"{Fore.YELLOW}Deck Sizes: {Fore.GREEN}Low: {len(self.deck_low)}, Medium: {len(self.deck_medium)}, High: {len(self.deck_high)}")
            print(f"\n{Fore.MAGENTA}🦹‍ Fugitive Hand: {Fore.WHITE}{', '.join(map(str, multibinary_to_cards(self.fugitive_hand)))}")
            print(f"{Fore.BLUE}🚔 Marshal Hand: {Fore.WHITE}{', '.join(map(str, multibinary_to_cards(self.marshal_hand)))}")
            print(f"{Fore.YELLOW}Last Hideout: {Fore.GREEN}{self.last_hideout}")
            print(f"{Fore.YELLOW}Is Manhunt: {Fore.GREEN}{self.is_manhunt}")

            print(f"\n{Fore.MAGENTA}Fugitive's View:")
            for i in range(self.n_hideouts+1):
                hideout = self.fugitive_center_row[i]
                sprint = ', '.join(map(str, multibinary_to_cards(self.fugitive_center_row_sprint[i])))
                print(f"  Card {i}: Hideout: {Fore.GREEN}{hideout}{Fore.MAGENTA}, Sprint: {Fore.GREEN}{sprint}")

            print(f"\n{Fore.BLUE}Marshal's View:")
            for i in range(self.n_hideouts+1):
                hideout = self.marshal_center_row[i]
                sprint_size = self.marshal_center_row_sprint_size[i]
                guessed = ', '.join(map(str, multibinary_to_cards(self.guessed_center_row[i,:])))
                possible_range = ', '.join(map(str, multibinary_to_cards(self.marshal_notes.get_hideouts_range()[i])))
                print(f"  Card {i}: Hideout: {Fore.GREEN}{hideout}{Fore.BLUE}, Sprint Size: {Fore.GREEN}{sprint_size}")
                print(f"    Guessed: {Fore.GREEN}{guessed}")
                print(f"    Possible Range: {Fore.GREEN}{possible_range}")

            print(f"\n{Fore.YELLOW}Game State:")
            print(f"  Guessed: {Fore.GREEN}{', '.join(map(str, multibinary_to_cards(self.guessed)))}")
            print(f"  Revealed Hideout: {Fore.GREEN}{', '.join(map(str, multibinary_to_cards(self.revealed_hideout)))}")
            print(f"  Revealed Sprint: {Fore.GREEN}{', '.join(map(str, multibinary_to_cards(self.revealed_sprint)))}")
            print(f"  Is 30 or above revealed: {Fore.GREEN}{self._is_30_or_above_revealed()}")
            print(f"{Fore.CYAN}{'='*60}\n")

    def close(self):
        pass



if __name__ == "__main__": 
    icons = {
        PlayerRole.FUGITIVE_DRAW: "🦹‍",
        PlayerRole.FUGITIVE_HIHDEOUT: "🦹‍",
        PlayerRole.FUGITIVE_SPRINT: "🦹‍",
        PlayerRole.MARSHAL_DRAW: "🚔", 
        PlayerRole.MARSHAL_GUESS: "🚔"
    }

    def action_verb(player_role):
        if player_role == PlayerRole.FUGITIVE_DRAW or player_role == PlayerRole.MARSHAL_DRAW:
            return "Draw from deck"
        if player_role == PlayerRole.FUGITIVE_HIHDEOUT:
            return "Play Hideout"
        if player_role == PlayerRole.FUGITIVE_SPRINT:
            return "Play Sprint"
        if player_role == PlayerRole.MARSHAL_GUESS:
            return "Guess"

    # Example usage
    from pettingzoo.test import api_test

    env = Fugitive(render_mode="human")
    # env.reset()
    # print(env.observation_space(env.agent_selection).sample())
    # api_test(env, num_cycles=1000, verbose_progress=False)

    # Run a few episodes
    for episode in range(1):
        env.reset()
        turn_count = 0
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            flattened_observation = flatten(env.observation_space(env.agent_selection), observation)
            if termination or truncation:
                action = None
                break
            else:
                mask = info["action_masks"]
                action = env.action_space(agent).sample(mask)
            env.render()
            print(f"Agent {icons[agent]}: Available action: {np.where(mask==1)[0]}")
            print(f"Agent {icons[agent]} decided to {action_verb(env.agent_selection)} {action}")
            env.step(action)
            turn_count += 1
        print(f"Turn Count: {turn_count}")
        print(f"Episode {episode + 1} finished. Rewards: {env.rewards}")
  