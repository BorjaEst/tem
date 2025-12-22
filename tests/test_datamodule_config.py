"""Test new DataModuleConfig API."""

import pytest

from torch_tem.config import DataModuleConfig, DistancePolicyConfig, EnvironmentConfig, MixedPolicyConfig, QLearningPolicyConfig, RandomPolicyConfig, ShinyPolicyConfig


def test_basic_config():
    """Test basic DataModuleConfig construction."""
    config = DataModuleConfig(
        environment=EnvironmentConfig(width=5, height=5),
        batch_size=16,
        sequence_length=64,
        policy=RandomPolicyConfig(walk_length_min=25, walk_length_max=100),
        n_train_batches=1000,
    )

    assert config.batch_size == 16
    assert config.sequence_length == 64
    assert config.policy.walk_length_min == 25
    assert config.policy.walk_length_max == 100
    assert config.environment.n_locations == 25
    assert isinstance(config.policy, RandomPolicyConfig)


def test_policy_configs():
    """Test different policy configurations."""
    # Random policy
    random_config = DataModuleConfig(
        environment=EnvironmentConfig(width=5, height=5),
        sequence_length=100,
        policy=RandomPolicyConfig(walk_length_min=50, walk_length_max=300),
        n_train_batches=100,
    )
    assert random_config.policy.type == "random"
    assert random_config.policy.walk_length_min == 50
    assert random_config.policy.walk_length_max == 300

    # Distance policy
    distance_config = DataModuleConfig(
        environment=EnvironmentConfig(width=5, height=5),
        sequence_length=80,
        policy=DistancePolicyConfig(
            beta=2.0,
            goal_mode="random",
            n_goals=1,
            walk_length_min=25,
            walk_length_max=100,
        ),
        n_train_batches=100,
    )
    assert distance_config.policy.type == "distance"
    assert distance_config.policy.beta == 2.0
    assert distance_config.policy.walk_length_min == 25

    # Q-learning policy
    qlearn_config = DataModuleConfig(
        environment=EnvironmentConfig(width=5, height=5),
        sequence_length=80,
        policy=QLearningPolicyConfig(
            gamma=0.9,
            beta=1.0,
            walk_length_min=25,
            walk_length_max=100,
        ),
        n_train_batches=100,
    )
    assert qlearn_config.policy.type == "q_learning"
    assert qlearn_config.policy.gamma == 0.9
    assert qlearn_config.policy.walk_length_max == 100


def test_mixed_policy():
    """Test mixed policy configuration."""
    config = DataModuleConfig(
        environment=EnvironmentConfig(width=5, height=5),
        sequence_length=120,
        policy=MixedPolicyConfig(
            policies=[
                RandomPolicyConfig(walk_length_min=50, walk_length_max=300),
                DistancePolicyConfig(beta=2.0, walk_length_min=25, walk_length_max=100),
            ],
            weights=[0.7, 0.3],
        ),
        n_train_batches=100,
    )

    assert config.policy.type == "mixed"
    assert len(config.policy.policies) == 2
    assert sum(config.policy.weights) == pytest.approx(1.0)
    # Policy walk-length helpers remain, but the DataModule uses sequence_length.
    assert config.sequence_length == 120


def test_mixed_policy_validation():
    """Test mixed policy weight validation."""
    with pytest.raises(ValueError, match="must sum to 1.0"):
        DataModuleConfig(
            environment=EnvironmentConfig(width=5, height=5),
            policy=MixedPolicyConfig(
                policies=[
                    RandomPolicyConfig(),
                    DistancePolicyConfig(),
                ],
                weights=[0.5, 0.6],  # Sum > 1.0
            ),
            n_train_batches=100,
        )

    with pytest.raises(ValueError, match="must match number of policies"):
        DataModuleConfig(
            environment=EnvironmentConfig(width=5, height=5),
            policy=MixedPolicyConfig(
                policies=[
                    RandomPolicyConfig(),
                    DistancePolicyConfig(),
                ],
                weights=[0.5],  # Too few weights
            ),
            n_train_batches=100,
        )


def test_walk_length_is_sole_source():
    """Walk length is fixed and comes only from DataModuleConfig.sequence_length."""
    config = DataModuleConfig(
        environment=EnvironmentConfig(width=5, height=5),
        sequence_length=77,
        policy=RandomPolicyConfig(
            walk_length_min=25,
            walk_length_max=300,
            walk_length_curriculum=True,
            walk_length_curriculum_steps=1000,
        ),
        n_train_batches=1000,
    )

    assert config.sequence_length == 77


def test_walk_length_ignores_policy_fields():
    """Policy walk-length fields do not affect the configured sequence length."""
    config = DataModuleConfig(
        environment=EnvironmentConfig(width=5, height=5),
        sequence_length=50,
        policy=RandomPolicyConfig(
            walk_length_min=10,
            walk_length_max=999,
            walk_length_curriculum=False,
        ),
        n_train_batches=1000,
    )

    assert config.sequence_length == 50


def test_shiny_integration():
    """Test shiny object configuration."""
    config = DataModuleConfig(
        environment=EnvironmentConfig(width=5, height=5),
        sequence_length=100,
        policy=ShinyPolicyConfig(
            navigation_policy="distance",
            beta=1.5,
            gamma=0.7,
            n=2,
            returns=15,
            min_separation=0.3,
        ),
        n_train_batches=100,
    )

    assert config.policy.type == "shiny"
    assert config.policy.n == 2
    assert config.policy.returns == 15
    assert config.policy.beta == 1.5
    assert config.policy.gamma == 0.7


def test_multi_environment_fields_are_rejected():
    """Multi-environment knobs were removed; extra fields must be rejected."""
    with pytest.raises(Exception):
        DataModuleConfig(
            environment=EnvironmentConfig(width=5, height=5),
            batch_size=16,
            sequence_length=100,
            n_environments=5,
            environments_per_batch=2,
            n_train_batches=100,
        )


def test_shiny_policy_config():
    """Test ShinyPolicyConfig construction and validation."""
    # Default values
    shiny = ShinyPolicyConfig()
    assert shiny.type == "shiny"
    assert shiny.navigation_policy == "distance"
    assert shiny.n == 2
    assert shiny.returns == 15
    assert shiny.beta == 1.5
    assert shiny.gamma == 0.7
    assert shiny.min_separation == 0.3

    # Custom values
    custom_shiny = ShinyPolicyConfig(
        navigation_policy="q_learning",
        beta=2.0,
        gamma=0.85,
        n=3,
        returns=20,
        min_separation=0.4,
    )
    assert custom_shiny.navigation_policy == "q_learning"
    assert custom_shiny.n == 3
    assert custom_shiny.returns == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
