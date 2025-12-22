"""Legacy configuration adapter for TEM model comparison.

This module provides conversion between the legacy dict-based parameter format
(from parameters.py) and the new typed configuration system (ModelConfig, etc.).

Critical conversions:
- Lambda/kappa inversion: legacy lambda (retention) → new kappa (forgetting = 1-lambda)
- Legacy kappa (retrieval decay) → new kappa (same name, same meaning)
- Batch memory handling: legacy batched [B, n_p, n_p] → new optionally batched
"""

from typing import Any, Dict

import torch
from torch import Tensor

from torch_tem.config import EnvironmentConfig, ModelConfig, TrainingConfig


class ConfigWrapper:
    """Wrapper combining architecture and inference configs for TEMModel.

    TEMModel expects params.architecture and params.inference attributes.
    """

    def __init__(self, architecture, inference, training=None, environment=None):
        self.architecture = architecture
        self.inference = inference
        self.training = training  # Optional, for training scripts
        self.environment = environment  # Optional, for world generation


def legacy_to_typed(params: Dict[str, Any]) -> ConfigWrapper:
    """Convert legacy parameter dict to typed ConfigWrapper.

    This function handles all the critical conversions needed to make the
    refactored TEMModel produce identical results to the legacy Model:

    1. **Lambda/Kappa Naming**:
       - Legacy `lambda` (0.9999) = memory retention rate
       - Refactored uses: lamb parameter in storage.update()
       - Legacy `kappa` (0.8) = retrieval decay (attractor dynamics)
       - Refactored `kappa` = same meaning (retrieval decay)

    2. **Memory Structure**:
       - Legacy: batched [batch, sum(n_p), sum(n_p)]
       - Refactored: batched if batch_size > 1, else global

    3. **Module Connectivity**:
       - Both use same hierarchical masks (computed from architecture)

    Args:
        params: Legacy parameter dictionary from parameters.parameters()

    Returns:
        ConfigWrapper with architecture, inference, training, environment configs

    Example:
        >>> import parameters
        >>> legacy_params = parameters.parameters()
        >>> config = legacy_to_typed(legacy_params)
        >>> # Now use with refactored model
        >>> from torch_tem.model import TEMModel
        >>> model = TEMModel(config)
        >>> model.set_batch_size(legacy_params['batch_size'])
    """
    # Extract base architectural parameters (also merges env/inf fields)
    architecture = _create_merged_architecture(params)

    # Extract training parameters (loss weights, learning rates, curriculum)
    training = _create_training_config(params)

    # Extract environment parameters (action space, shiny objects)
    environment = _create_environment_config(params)

    # Attach training and environment configs to the architecture object
    # This allows access to all config types from the single object passed to TEMModel
    architecture.training = training
    architecture.environment = environment

    # Return the merged configuration object directly
    # TEMModel expects an object with direct access to architecture fields (n_g, n_x, etc.)
    return architecture


def _create_merged_architecture(params: Dict[str, Any]):
    """Create merged architecture config with environment and inference fields.

    TEMModel components need fields from multiple configs, so we merge them
    into the architecture object that gets passed around."""
    from torch_tem.config import ModelConfig

    # Create base architecture config with all required fields
    config = ModelConfig(
        # Base dimensions
        batch_size=params["batch_size"],
        n_x=params["n_x"],
        n_x_c=params["n_x_c"],
        n_g_subsampled=params["n_g_subsampled"][: params["n_f_g"]],  # Exclude OVC if separate
        n_ovc=params.get("n_ovc", []),
        f_initial=params["f_initial"][: len(params["n_g_subsampled"]) - params.get("n_f_ovc", 0)],
        separate_ovc=params.get("separate_ovc", False),
        # Network initialization
        g_init_std=params["g_init_std"],
        g_mem_std=params["g_mem_std"],
        d_hidden_dim=params["d_hidden_dim"],
        n_actions=params["n_actions"],
        # Memory structure
        common_memory=params["common_memory"],
        # Training parameters
        eta=params["eta"],
    )

    return ConfigWrapper(
        architecture=config, inference=config, training=_create_training_config(params), environment=_create_environment_config(params)  # Share the same config object
    )


# InferenceConfig has been merged into ModelConfig
# Inference parameters (do_sample, use_p_inf, eta, kappa) are now part of ModelConfig


def _create_training_config(params: Dict[str, Any]) -> TrainingConfig:
    """Extract training configuration from legacy params."""
    return TrainingConfig(
        # Training schedule
        n_rollout=params["n_rollout"],
        # Loss weights (base values before curriculum)
        loss_weights_x=params["loss_weights_x"],
        loss_weights_p=params["loss_weights_p"],
        loss_weights_g=params["loss_weights_g"],
        loss_weights_reg_g=params["loss_weights_reg_g"],
        loss_weights_reg_p=params["loss_weights_reg_p"],
        # Loss weight curriculum schedule
        loss_weights_p_g_it=params["loss_weights_p_g_it"],
        loss_weights_reg_p_it=params["loss_weights_reg_p_it"],
        loss_weights_reg_g_it=params["loss_weights_reg_g_it"],
        # Memory curriculum schedule
        eta_it=params["eta_it"],
        lambda_it=params["lambda_it"],
        # Learning rate schedule
        lr_max=params["lr_max"],
        lr_min=params["lr_min"],
        lr_decay_rate=params["lr_decay_rate"],
        lr_decay_steps=params["lr_decay_steps"],
        # Precision weighting schedule (p→g inference)
        p2g_scale_offset=params.get("p2g_scale_offset", 0.0),
        p2g_sig_val=params.get("p2g_sig_val", 10000.0),
        p2g_sig_half_it=params.get("p2g_sig_half_it", 400),
        p2g_sig_scale_it=params.get("p2g_sig_scale_it", 200),
    )


def _create_environment_config(params: Dict[str, Any]) -> EnvironmentConfig:
    """Extract environment configuration from legacy params."""
    return EnvironmentConfig(
        # Action space
        n_actions=params["n_actions"],
        has_static_action=params["has_static_action"],
        explore_bias=params["explore_bias"],
        # Shiny objects (optional)
        shiny_rate=params.get("shiny_rate", 0),
        shiny_gamma=params.get("shiny_gamma", 0.7),
        shiny_beta=params.get("shiny_beta", 1.5),
        shiny_n=params.get("shiny_n", 2),
        shiny_returns=params.get("shiny_returns", 15),
    )


def get_memory_parameters(params: Dict[str, Any], iteration: int = 0) -> Dict[str, float]:
    """Get memory parameters (eta, lambda) with curriculum scheduling.

    This replicates the parameter_iteration() logic from legacy parameters.py
    to compute time-varying memory parameters during training.

    Args:
        params: Legacy parameter dictionary
        iteration: Current training iteration (0-indexed)

    Returns:
        Dictionary with 'eta' (remembering) and 'lamb' (forgetting) values

    Example:
        >>> params = parameters.parameters()
        >>> memory_params = get_memory_parameters(params, iteration=1000)
        >>> model.storage.update(p_inf, p_gen, **memory_params)
    """
    # Calculate eta (rate of remembering) with curriculum
    eta = min((iteration + 1) / params["eta_it"], 1.0) * params["eta"]

    # Calculate lambda (rate of forgetting) with curriculum
    # Legacy lambda is retention, so we use it directly
    lamb = min((iteration + 1) / params["lambda_it"], 1.0) * params["lambda"]

    return {"eta": eta, "lamb": lamb}


def get_loss_weights(params: Dict[str, Any], iteration: int = 0) -> Tensor:
    """Get loss weights with curriculum scheduling.

    Replicates the loss weight computation from parameter_iteration() in
    legacy parameters.py, applying curriculum schedules.

    Args:
        params: Legacy parameter dictionary
        iteration: Current training iteration (0-indexed)

    Returns:
        Tensor of shape [8] with loss weights in order:
        [L_p_g, L_p_x, L_x_gen, L_x_g, L_x_p, L_g, L_reg_g, L_reg_p]

    Example:
        >>> params = parameters.parameters()
        >>> weights = get_loss_weights(params, iteration=5000)
        >>> loss = torch.sum(weights * torch.stack([...losses...]))
    """
    import numpy as np

    # Calculate p2g scaling offset (for precision weighting)
    p2g_scale_offset = 1 / (1 + np.exp((iteration - params["p2g_sig_half_it"]) / params["p2g_sig_scale_it"]))

    # Calculate curriculum-scheduled loss weights
    L_p_g = min((iteration + 1) / params["loss_weights_p_g_it"], 1.0) * params["loss_weights_p"]
    L_p_x = min((iteration + 1) / params["loss_weights_p_g_it"], 1.0) * params["loss_weights_p"] * (1 - p2g_scale_offset)
    L_x_gen = params["loss_weights_x"]
    L_x_g = params["loss_weights_x"]
    L_x_p = params["loss_weights_x"]
    L_g = min((iteration + 1) / params["loss_weights_p_g_it"], 1.0) * params["loss_weights_g"]
    L_reg_g = (1 - min((iteration + 1) / params["loss_weights_reg_g_it"], 1.0)) * params["loss_weights_reg_g"]
    L_reg_p = (1 - min((iteration + 1) / params["loss_weights_reg_p_it"], 1.0)) * params["loss_weights_reg_p"]

    return torch.tensor([L_p_g, L_p_x, L_x_gen, L_x_g, L_x_p, L_g, L_reg_g, L_reg_p], dtype=torch.float)


def get_learning_rate(params: Dict[str, Any], iteration: int = 0) -> float:
    """Get learning rate with exponential decay schedule.

    Args:
        params: Legacy parameter dictionary
        iteration: Current training iteration (0-indexed)

    Returns:
        Current learning rate
    """
    lr = params["lr_max"] * (params["lr_decay_rate"] ** (iteration / params["lr_decay_steps"]))
    return max(lr, params["lr_min"])


def get_walk_length_center(params: Dict[str, Any], iteration: int = 0) -> float:
    """Get center of walk length sampling window with curriculum.

    Args:
        params: Legacy parameter dictionary
        iteration: Current training iteration (0-indexed)

    Returns:
        Center of walk length window for uniform sampling
    """
    progress = min((iteration + 1) / params["train_it"], 1.0)
    walk_length_range = params["walk_it_max"] - params["walk_it_min"] - params["walk_it_window"]
    return params["walk_it_max"] - params["walk_it_window"] * 0.5 - progress * walk_length_range


# ======================================================================================
# USAGE EXAMPLE
# ======================================================================================

if __name__ == "__main__":
    """Example demonstrating legacy-to-typed configuration conversion."""
    import os
    import sys

    # Add parent directory to path to import legacy modules
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import parameters
    from model import Model as LegacyModel
    from torch_tem.core.model import TEMModel

    print("=" * 70)
    print("LEGACY TO TYPED CONFIGURATION CONVERSION")
    print("=" * 70)

    # Load legacy parameters
    print("\n1. Loading legacy parameters...")
    legacy_params = parameters.parameters()
    print(f"   - Legacy lambda (retention): {legacy_params['lambda']}")
    print(f"   - Legacy kappa (retrieval): {legacy_params['kappa']}")
    print(f"   - Legacy eta (remembering): {legacy_params['eta']}")
    print(f"   - Batch size: {legacy_params['batch_size']}")

    # Convert to typed config
    print("\n2. Converting to typed configuration...")
    typed_config = legacy_to_typed(legacy_params)
    print(f"   - Architecture.n_f: {typed_config.architecture.n_f}")
    print(f"   - Architecture.n_g: {typed_config.architecture.n_g}")
    print(f"   - Architecture.n_p: {typed_config.architecture.n_p}")
    print(f"   - Inference.kappa: {typed_config.inference.kappa}")
    print(f"   - Inference.eta: {typed_config.inference.eta}")

    # Create models
    print("\n3. Creating models...")
    from model import Model as LegacyModel

    legacy_model = LegacyModel(legacy_params)
    refactored_model = TEMModel(typed_config)
    refactored_model.set_batch_size(legacy_params["batch_size"])

    print(f"   - Legacy model parameters: {sum(p.numel() for p in legacy_model.parameters())}")
    print(f"   - Refactored model parameters: {sum(p.numel() for p in refactored_model.parameters())}")

    # Get curriculum-scheduled parameters
    print("\n4. Testing curriculum scheduling (iteration 1000)...")
    memory_params = get_memory_parameters(legacy_params, iteration=1000)
    loss_weights = get_loss_weights(legacy_params, iteration=1000)
    lr = get_learning_rate(legacy_params, iteration=1000)

    print(f"   - Scheduled eta: {memory_params['eta']:.6f}")
    print(f"   - Scheduled lambda: {memory_params['lamb']:.6f}")
    print(f"   - Learning rate: {lr:.6e}")
    print(f"   - Loss weights shape: {loss_weights.shape}")

    print("\n" + "=" * 70)
    print("✓ Conversion successful!")
    print("=" * 70)
