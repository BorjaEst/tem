"""Environment JSON validation against space contract.

This module provides strict validation of environment JSON files to ensure they
match the model's expected observation and action space dimensions before training.

Key Features:
    - Fast-fail validation at startup (before trainer/model initialization)
    - Actionable error messages with file path and context
    - Structural sanity checks (required keys, dimension consistency)
    - Action 0 no-op enforcement when enabled by contract
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch_tem.settings import SpaceContractSettings


class EnvironmentValidationError(ValueError):
    """Raised when environment JSON violates space contract."""

    pass


def validate_envs_against_contract(
    env_paths: list[Path], contract: SpaceContractSettings
) -> None:
    """Validate all environment JSON files against the space contract.

    Performs strict validation to ensure:
        - All required keys are present
        - n_observations and n_actions match contract
        - Observation IDs are within valid range
        - Action 0 is a no-op when required
        - Structural invariants hold (location count, transition lengths, etc.)

    Args:
        env_paths: List of environment JSON file paths to validate.
        contract: Space contract settings defining expected dimensions.

    Raises:
        EnvironmentValidationError: If any validation check fails.
            Error message includes file path, specific check that failed,
            expected vs actual values, and location context where relevant.

    Example:
        >>> from pathlib import Path
        >>> from torch_tem.settings import SpaceContractSettings
        >>> contract = SpaceContractSettings(n_observations=45, n_actions_move=4)
        >>> validate_envs_against_contract([Path("env.json")], contract)
    """
    if not env_paths:
        raise EnvironmentValidationError("No environment files provided for validation")

    for env_path in env_paths:
        _validate_single_env(env_path, contract)


def _validate_single_env(env_path: Path, contract: SpaceContractSettings) -> None:
    """Validate a single environment JSON file against contract.

    Args:
        env_path: Path to environment JSON file.
        contract: Space contract settings.

    Raises:
        EnvironmentValidationError: If validation fails.
    """
    # Step 1: File existence and JSON parsing
    if not env_path.exists():
        raise EnvironmentValidationError(
            f"Environment file not found: {env_path}"
        )

    try:
        with open(env_path, "r") as f:
            env = json.load(f)
    except json.JSONDecodeError as e:
        raise EnvironmentValidationError(
            f"Invalid JSON in {env_path}: {e}"
        ) from e

    # Step 2: Required keys
    required_keys = ["n_observations", "n_actions", "n_locations", "locations", "adjacency"]
    missing_keys = [key for key in required_keys if key not in env]
    if missing_keys:
        raise EnvironmentValidationError(
            f"Environment {env_path} missing required keys: {missing_keys}"
        )

    # Step 3: Contract dimension match
    if env["n_observations"] != contract.n_observations:
        raise EnvironmentValidationError(
            f"Environment {env_path} n_observations mismatch: "
            f"expected {contract.n_observations}, got {env['n_observations']}"
        )

    if env["n_actions"] != contract.n_actions_total:
        raise EnvironmentValidationError(
            f"Environment {env_path} n_actions mismatch: "
            f"expected {contract.n_actions_total} (n_actions_move={contract.n_actions_move}, "
            f"action0_is_noop={contract.action0_is_noop}), got {env['n_actions']}"
        )

    # Step 4: Structural sanity
    n_locations = env["n_locations"]
    locations = env["locations"]

    if len(locations) != n_locations:
        raise EnvironmentValidationError(
            f"Environment {env_path} location count mismatch: "
            f"n_locations={n_locations} but len(locations)={len(locations)}"
        )

    adjacency = env["adjacency"]
    if len(adjacency) != n_locations:
        raise EnvironmentValidationError(
            f"Environment {env_path} adjacency matrix row count mismatch: "
            f"expected {n_locations}, got {len(adjacency)}"
        )

    for i, row in enumerate(adjacency):
        if len(row) != n_locations:
            raise EnvironmentValidationError(
                f"Environment {env_path} adjacency matrix row {i} column count mismatch: "
                f"expected {n_locations}, got {len(row)}"
            )

    # Step 5: Location-level validation
    for loc in locations:
        loc_id = loc.get("id")
        
        # Validate observation ID range
        obs_id = loc.get("observation")
        if obs_id is None:
            raise EnvironmentValidationError(
                f"Environment {env_path} location {loc_id} missing 'observation' field"
            )
        if not (0 <= obs_id < contract.n_observations):
            raise EnvironmentValidationError(
                f"Environment {env_path} location {loc_id} has observation ID {obs_id} "
                f"outside valid range [0, {contract.n_observations})"
            )

        # Validate actions structure
        actions = loc.get("actions")
        if actions is None:
            raise EnvironmentValidationError(
                f"Environment {env_path} location {loc_id} missing 'actions' field"
            )
        if len(actions) != env["n_actions"]:
            raise EnvironmentValidationError(
                f"Environment {env_path} location {loc_id} has {len(actions)} actions, "
                f"expected {env['n_actions']}"
            )

        # Validate action 0 no-op rule
        if contract.action0_is_noop:
            action0 = actions[0]
            transition = action0.get("transition")
            if transition is None:
                raise EnvironmentValidationError(
                    f"Environment {env_path} location {loc_id} action 0 missing 'transition' field"
                )
            if len(transition) != n_locations:
                raise EnvironmentValidationError(
                    f"Environment {env_path} location {loc_id} action 0 transition length mismatch: "
                    f"expected {n_locations}, got {len(transition)}"
                )

            # Check for strict self-loop: transition[loc_id] == 1, all others == 0
            if loc_id is not None:
                if transition[loc_id] != 1:
                    raise EnvironmentValidationError(
                        f"Environment {env_path} location {loc_id} action 0 is not a self-loop: "
                        f"transition[{loc_id}] = {transition[loc_id]} (expected 1)"
                    )
                for i, prob in enumerate(transition):
                    if i != loc_id and prob != 0:
                        raise EnvironmentValidationError(
                            f"Environment {env_path} location {loc_id} action 0 is not a no-op: "
                            f"transition[{i}] = {prob} (expected 0 for all i != {loc_id})"
                        )

        # Validate all action transition lengths
        for action_idx, action in enumerate(actions):
            transition = action.get("transition")
            if transition is None:
                raise EnvironmentValidationError(
                    f"Environment {env_path} location {loc_id} action {action_idx} "
                    f"missing 'transition' field"
                )
            if len(transition) != n_locations:
                raise EnvironmentValidationError(
                    f"Environment {env_path} location {loc_id} action {action_idx} "
                    f"transition length mismatch: expected {n_locations}, got {len(transition)}"
                )
