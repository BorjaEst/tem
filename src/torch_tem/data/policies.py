"""Policy generation for navigation and goal-directed behavior."""

import copy
from typing import List, Union

import numpy as np

from torch_tem.data.environment import Environment, Location


class PolicyGenerator:
    """Generates action policies for environment navigation.

    Supports multiple policy types for different training phases:
    - Random exploration (uniform over valid actions)
    - Distance-based goal policies (softmax over graph distances)
    - Q-learning policies (value iteration toward reward locations)
    - Policy mixing for curriculum learning
    """

    def __init__(self, environment: Environment):
        """Initialize policy generator for given environment.

        Args:
            environment: Environment instance to generate policies for
        """
        self.env = environment
        self.distance_cache = environment.shortest_paths()

    def random_policy(self) -> List[Location]:
        """Generate uniform distribution over valid actions at each location.

        Returns:
            List[Location]: Locations with uniform action probabilities
        """
        new_locations = []

        for location in self.env.locations:
            # Count available actions (those with non-zero transitions)
            available_actions = [action for action in location.actions if sum(action.transition) > 0]
            n_available = len(available_actions)

            if n_available == 0:
                # No available actions, keep original probabilities
                new_locations.append(location)
                continue

            # Set uniform probability for available actions
            uniform_prob = 1.0 / n_available
            new_actions = []

            for action in location.actions:
                if sum(action.transition) > 0:
                    new_action = action.model_copy(update={"probability": uniform_prob})
                else:
                    new_action = action.model_copy(update={"probability": 0.0})
                new_actions.append(new_action)

            new_location = location.model_copy(update={"actions": new_actions})
            new_locations.append(new_location)

        return new_locations

    def distance_policy(self, goal_locations: Union[int, List[int]], beta: float = 1.0) -> List[Location]:
        """Softmax policy based on graph distance to goals.

        Computes policy that favors actions leading toward goal locations
        using softmax over negative distances. Faster than Q-learning but
        ignores transition probabilities.

        Args:
            goal_locations: Target location(s) to approach
            beta: Inverse temperature for softmax (higher = more deterministic)

        Returns:
            List[Location]: Locations with distance-based action probabilities
        """
        # Ensure goal_locations is a list
        if isinstance(goal_locations, int):
            goal_locations = [goal_locations]

        # Copy locations for modification
        new_locations = copy.deepcopy(self.env.locations)

        # Create boolean mask for goal locations
        is_goal = np.zeros(self.env.n_locations, dtype=bool)
        for goal in goal_locations:
            is_goal[goal] = True

        # Disable self-actions at goal locations (they become too attractive)
        for goal_id in goal_locations:
            for action in new_locations[goal_id].actions:
                if action.transition[goal_id] == 1.0:
                    action.probability = 0.0

            # Renormalize remaining actions
            total_prob = sum(a.probability for a in new_locations[goal_id].actions)
            if total_prob > 0:
                for action in new_locations[goal_id].actions:
                    action.probability = action.probability / total_prob

        # Compute minimum distance to any goal for each action
        location_action_distances = []

        for location in new_locations:
            action_distances = []
            for action in location.actions:
                # Find states reachable by this action
                reachable = np.array(action.transition) > 0

                if not any(reachable):
                    action_distances.append(np.inf)
                else:
                    # Minimum distance from any goal to any reachable state
                    distances = self.distance_cache[is_goal][:, reachable]
                    action_distances.append(np.min(distances) if distances.size > 0 else np.inf)
            location_action_distances.append(action_distances)

        # Calculate policy from softmax over negative distances
        for location, action_distances in zip(new_locations, location_action_distances):
            # Get distances for available actions
            distances = np.array([-d if location.actions[i].probability > 0 else -np.inf for i, d in enumerate(action_distances)])

            # Softmax with temperature beta
            exp_values = np.exp(beta * distances)
            probabilities = exp_values / np.sum(exp_values)

            # Update action probabilities
            for action, prob in zip(location.actions, probabilities):
                action.probability = float(prob)

        # Convert back to Pydantic models
        return [Location(**loc.model_dump()) for loc in new_locations]

    def q_learning_policy(self, goal_locations: Union[int, List[int]], gamma: float = 0.9, beta: float = 1.0, n_iterations: int = 100) -> List[Location]:
        """Q-learned policy via value iteration.

        Computes optimal policy toward reward locations using value iteration
        on the Bellman equation. Accounts for transition probabilities.

        Args:
            goal_locations: Reward location(s)
            gamma: Discount factor for future rewards
            beta: Softmax temperature
            n_iterations: Value iteration steps (default 10 * n_locations)

        Returns:
            List[Location]: Locations with Q-learned action probabilities
        """
        # Ensure goal_locations is a list
        if isinstance(goal_locations, int):
            goal_locations = [goal_locations]

        # Copy locations for modification
        new_locations = copy.deepcopy(self.env.locations)

        # Disable self-actions at goal locations
        for goal_id in goal_locations:
            for action in new_locations[goal_id].actions:
                if action.transition[goal_id] == 1.0:
                    action.probability = 0.0

            # Renormalize
            total_prob = sum(a.probability for a in new_locations[goal_id].actions)
            if total_prob > 0:
                for action in new_locations[goal_id].actions:
                    action.probability = action.probability / total_prob

        # Initialize Q-values to 0
        q_values = [[0.0 for _ in location.actions] for location in new_locations]

        # Value iteration
        if n_iterations is None:
            n_iterations = 10 * self.env.n_locations

        for _ in range(n_iterations):
            # Deep copy Q-values for synchronous update
            prev_q_values = [list(q_vals) for q_vals in q_values]

            for loc_id, location in enumerate(new_locations):
                for action_id, action in enumerate(location.actions):
                    # Bellman update: Q(s,a) = Σ p(s'|s,a) * [r(s') + γ * max_a' Q(s',a')]
                    q_value = 0.0
                    for next_loc_id, trans_prob in enumerate(action.transition):
                        if trans_prob > 0:
                            # Reward is 1 if next state is goal, 0 otherwise
                            reward = 1.0 if next_loc_id in goal_locations else 0.0

                            # Max Q-value over actions at next state
                            max_q_next = max(prev_q_values[next_loc_id])

                            q_value += trans_prob * (reward + gamma * max_q_next)

                    q_values[loc_id][action_id] = q_value

        # Calculate policy from softmax over Q-values
        for loc_id, location in enumerate(new_locations):
            q_vals_array = np.array([q_values[loc_id][i] if action.probability > 0 else -np.inf for i, action in enumerate(location.actions)])

            # Softmax
            exp_values = np.exp(beta * q_vals_array)
            probabilities = exp_values / np.sum(exp_values)

            # Update probabilities
            for action, prob in zip(location.actions, probabilities):
                action.probability = float(prob)

        # Convert back to Pydantic models
        result = []
        for loc in new_locations:
            loc_dict = loc.model_dump()
            result.append(Location(**loc_dict))

        return result

    def mix_policies(self, policies: List[List[Location]], weights: List[float]) -> List[Location]:
        """Combine multiple policies with weighted averaging.

        Args:
            policies: List of policy location lists
            weights: Mixing weights (must sum to 1.0)

        Returns:
            List[Location]: Mixed policy

        Raises:
            ValueError: If policies have different lengths or weights don't sum to 1
        """
        if not policies:
            raise ValueError("At least one policy required")

        if len(policies) != len(weights):
            raise ValueError("Number of policies must match number of weights")

        if not (0.999 <= sum(weights) <= 1.001):
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")

        # Check all policies have same number of locations
        n_locs = len(policies[0])
        if not all(len(p) == n_locs for p in policies):
            raise ValueError("All policies must have same number of locations")

        mixed_locations = []

        for loc_id in range(n_locs):
            # Get corresponding location from each policy
            location_variants = [policy[loc_id] for policy in policies]

            # Mix action probabilities
            n_actions = len(location_variants[0].actions)
            mixed_actions = []

            for action_id in range(n_actions):
                # Weighted average of probabilities
                mixed_prob = sum(weight * loc.actions[action_id].probability for weight, loc in zip(weights, location_variants))

                # Use action from first policy as template
                template_action = location_variants[0].actions[action_id]
                mixed_action = template_action.model_copy(update={"probability": mixed_prob})
                mixed_actions.append(mixed_action)

            # Create mixed location
            mixed_location = location_variants[0].model_copy(update={"actions": mixed_actions})
            mixed_locations.append(mixed_location)

        return mixed_locations
