#!/usr/bin/env python3
import random
import math
import time

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        """
        # Generate game tree object
        first_msg = self.receiver()

        while True:
            msg = self.receiver()
            # Create the root node of the game tree
            initial_tree_node = Node(message=msg, player=0)

            # Find the best move using iterative deepening and move ordering
            best_move = self.search_best_next_move(initial_tree_node)
            
            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def calculate_heuristics(self, node):
        """
        Compute heuristic value for a given node.
        Combines:
          1. Difference in player scores.
          2. Proximity of the green hook to valuable fish.
          3. Penalizing proximity of red hook to valuable fish.
        """
        green_score = node.state.player_scores[0]
        red_score = node.state.player_scores[1]
        score_diff = green_score - red_score  # Prioritize maximizing score difference

        # Fish proximity scoring
        max_heuristic = 0
        for fish_id, fish_pos in node.state.get_fish_positions().items():
            fish_score = node.state.get_fish_scores()[fish_id]
            dist_green = self.l1_distance(fish_pos, node.state.get_hook_positions()[0])
            dist_red = self.l1_distance(fish_pos, node.state.get_hook_positions()[1])

            # Reward green catching fish and penalize red catching them
            if dist_green == 0 and fish_score > 0:
                return float("inf")  # Winning move
            if dist_red == 0:
                return float("-inf")  # Losing move

            # Weighted proximity heuristic
            max_heuristic = max(
                max_heuristic,
                fish_score * math.exp(-dist_green) - 0.5 * fish_score * math.exp(-dist_red)
            )

        return score_diff + max_heuristic

    def l1_distance(self, fish_positions, hook_positions):
        """
        Computes the Manhattan distance between the player hook and a given fish.
        """
        horizontal_dist = abs(fish_positions[0] - hook_positions[0])
        wrapped_horizontal = min(horizontal_dist, 20 - horizontal_dist)  # Handle wrapping
        vertical_dist = abs(fish_positions[1] - hook_positions[1])
        return wrapped_horizontal + vertical_dist

    def alphabeta(self, current_node, current_state, remaining_depth, lower_bound, upper_bound, active_player, start_time, visited_states):
        """
        Alpha-beta pruning with move ordering and timeout.
        """
        if time.time() - start_time > 0.05:
            raise TimeoutError

        state_key = self.hash_key(current_state)
        if state_key in visited_states and visited_states[state_key][0] >= remaining_depth:
            return visited_states[state_key][1]

        child_nodes = current_node.compute_and_get_children()
        child_nodes.sort(key=self.calculate_heuristics, reverse=True)  # Move ordering

        if remaining_depth == 0 or len(child_nodes) == 0:
            node_value = self.calculate_heuristics(current_node)
        elif active_player == 0:  # Maximizing player (green)
            node_value = float('-inf')
            for child_node in child_nodes:
                node_value = max(node_value, self.alphabeta(
                    child_node, child_node.state, remaining_depth - 1,
                    lower_bound, upper_bound, 1, start_time, visited_states))
                lower_bound = max(lower_bound, node_value)
                if lower_bound >= upper_bound:
                    break
        else:  # Minimizing player (red)
            node_value = float('inf')
            for child_node in child_nodes:
                node_value = min(node_value, self.alphabeta(
                    child_node, child_node.state, remaining_depth - 1,
                    lower_bound, upper_bound, 0, start_time, visited_states))
                upper_bound = min(upper_bound, node_value)
                if upper_bound <= lower_bound:
                    break

        visited_states[state_key] = [remaining_depth, node_value]
        return node_value

    def hash_key(self, state):
        """
        Computes a unique hash key for the game state based on hook and fish positions.
        """
        fish_data = {
            f"{position[0]}{position[1]}": score
            for (fish_id, position), (_, score) in zip(
                state.get_fish_positions().items(), state.get_fish_scores().items()
            )
        }
        return f"{state.get_hook_positions()}-{fish_data}"

    def depth_search(self, node, depth, initial_time, visited_states):
        """
        Performs alpha-beta pruning for a specific depth.
        """
        alpha = float('-inf')
        beta = float('inf')
        child_nodes = node.compute_and_get_children()
        scores = []

        for child_node in child_nodes:
            score = self.alphabeta(child_node, child_node.state, depth, alpha, beta, 1, initial_time, visited_states)
            scores.append(score)

        best_score_idx = scores.index(max(scores))
        return child_nodes[best_score_idx].move

    def search_best_next_move(self, initial_tree_node):
        """
        Iteratively deepens the search to find the best possible move.
        """
        initial_time = time.time()
        depth = 0
        timeout = False
        visited_states = dict()
        best_move = 0

        while not timeout:
            try:
                move = self.depth_search(initial_tree_node, depth, initial_time, visited_states)
                depth += 1
                best_move = move
            except TimeoutError:
                timeout = True

        return ACTION_TO_STR[best_move]
