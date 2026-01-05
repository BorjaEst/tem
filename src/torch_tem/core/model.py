#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:26:32 2020

This is a pytorch implementation of the Tolman-Eichenbaum Machine,
written by Jacob Bakermans after the original by James Whittington.
The referenced paper is the bioRxiv publication at https://www.biorxiv.org/content/10.1101/770495v2

Release v1.0.0: Fully functional pytorch model, without any extensions

@author: jacobb
"""
from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TypeAlias

# Standard modules
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, computed_field
from scipy.special import comb
from scipy.stats import truncnorm
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.modules import MLP


class Parameters(BaseModel):
    """
    Pydantic model for Tolman-Eichenbaum Machine model parameters.

    This model defines model architecture and inference parameters only.
    Training schedules and data generation settings are in torch_tem.settings.
    """

    # -- World/action parameters
    has_static_action: bool = Field(default=True, description="Does this world include the standing still action?")
    n_actions: int = Field(default=4, description="Number of available actions, excluding the stand still action")

    # -- Inference parameters
    p2g_sig_val: float = Field(default=10000.0, description="Additional value to offset standard deviation of inferred grounded location")

    # -- Model parameters
    do_sample: bool = Field(default=False, description="Whether to sample, or assume no noise and simply take mean of all distributions")
    use_p_inf: bool = Field(default=True, description="Whether to use inferred ground location while inferring new abstract location")
    separate_ovc: bool = Field(default=False, description="Whether to use separate grid modules that receive shiny information for object vector cells")
    g_init_std: float = Field(default=0.5, description="Standard deviation for initial g (which will then be learned)")
    g_mem_std: float = Field(default=0.1, description="Standard deviation to initialise hidden to output layer of MLP for inferring new abstract location")
    d_hidden_dim: int = Field(default=20, description="Hidden layer size of MLP for abstract location transitions")

    # ---- Neuron and module parameters
    n_g_subsampled_base: list[int] = Field(default=[10, 10, 8, 6, 6], description="Base neurons for subsampled entorhinal abstract location f_g(g) for each frequency module")
    n_ovc_base: list[int] | None = Field(default=None, description="Neurons for object vector cells")

    @computed_field
    @property
    def n_ovc(self) -> list[int]:
        """Neurons for object vector cells."""
        if self.n_ovc_base is not None:
            return self.n_ovc_base
        return [0 for _ in range(len(self.n_g_subsampled_base))]

    @computed_field
    @property
    def n_g_subsampled(self) -> list[int]:
        """
        Add neurons for object vector cells. Add new modules if object vector cells get separate modules,
        or else add neurons to existing modules.
        """
        if self.separate_ovc:
            return self.n_g_subsampled_base + self.n_ovc
        else:
            return [grid + ovc for grid, ovc in zip(self.n_g_subsampled_base, self.n_ovc)]

    @computed_field
    @property
    def n_f_ovc(self) -> int:
        """Number of hierarchical frequency modules for object vector cells."""
        return len(self.n_ovc) if self.separate_ovc else 0

    @computed_field
    @property
    def n_f_g(self) -> int:
        """Number of hierarchical frequency modules for grid cells."""
        return len(self.n_g_subsampled) - self.n_f_ovc

    @computed_field
    @property
    def n_f(self) -> int:
        """Total number of modules."""
        return len(self.n_g_subsampled)

    @computed_field
    @property
    def n_g(self) -> list[int]:
        """Number of neurons of entorhinal abstract location g for each frequency."""
        return [3 * g for g in self.n_g_subsampled]

    n_x: int = Field(default=45, description="Neurons for sensory observation x")
    n_x_c: int = Field(default=10, description="Neurons for compressed sensory experience x_c")

    @computed_field
    @property
    def n_x_f(self) -> list[int]:
        """Neurons for temporally filtered sensory experience x for each frequency."""
        return [self.n_x_c for _ in range(self.n_f)]

    @computed_field
    @property
    def n_p(self) -> list[int]:
        """Neurons for hippocampal grounded location p for each frequency."""
        return [g * x for g, x in zip(self.n_g_subsampled, self.n_x_f)]

    f_initial_base: list[float] = Field(default=[0.99, 0.3, 0.09, 0.03, 0.01], description="Initial frequencies of each module")

    @computed_field
    @property
    def f_initial(self) -> list[float]:
        """Initial frequencies of each module, including object vector cell modules."""
        return self.f_initial_base + self.f_initial_base[0 : self.n_f_ovc]

    # ---- Memory parameters
    common_memory: bool = Field(default=False, description="Use common memory for generative and inference network")
    kappa: float = Field(default=0.8, description="Hebbian retrieval decay term")

    @computed_field
    @property
    def i_attractor(self) -> int:
        """Number of iterations of attractor dynamics for memory retrieval."""
        return self.n_f_g

    @computed_field
    @property
    def i_attractor_max_freq_inf(self) -> list[int]:
        """Maximum iterations of attractor dynamics per frequency in inference model."""
        return [self.i_attractor for _ in range(self.n_f)]

    @computed_field
    @property
    def i_attractor_max_freq_gen(self) -> list[int]:
        """Maximum iterations of attractor dynamics per frequency in generative model."""
        return [self.i_attractor - freq_nr for freq_nr in range(self.n_f_g)] + [self.i_attractor for _ in range(self.n_f_ovc)]

    # --- Connectivity matrices
    @computed_field
    @property
    def p_update_mask(self) -> torch.Tensor:
        """
        Set connections when forming Hebbian memory of grounded locations: from low frequency modules to high.
        High frequency modules come first.
        """
        mask = torch.zeros((np.sum(self.n_p), np.sum(self.n_p)), dtype=torch.float)
        n_p = np.cumsum(np.concatenate(([0], self.n_p)))

        # Entry M_ij (row i, col j) is the connection FROM cell i TO cell j
        for f_from in range(self.n_f):
            for f_to in range(self.n_f):
                # For connections that involve separate object vector modules
                if f_from > self.n_f_g or f_to > self.n_f_g:
                    # Connection between object vector modules: only allow from low to high frequency
                    if f_from > self.n_f_g and f_to > self.n_f_g:
                        if self.f_initial[f_from] <= self.f_initial[f_to]:
                            mask[n_p[f_from] : n_p[f_from + 1], n_p[f_to] : n_p[f_to + 1]] = 1.0
                    # Connection between object vector and normal modules: allow any connections
                    else:
                        mask[n_p[f_from] : n_p[f_from + 1], n_p[f_to] : n_p[f_to + 1]] = 1.0
                # Connection between abstract location frequency modules: only from low to high frequency
                else:
                    if self.f_initial[f_from] <= self.f_initial[f_to]:
                        mask[n_p[f_from] : n_p[f_from + 1], n_p[f_to] : n_p[f_to + 1]] = 1.0

        return mask

    @computed_field
    @property
    def p_retrieve_mask_inf(self) -> list[torch.Tensor]:
        """Hierarchical memory retrieval masks for inference model."""
        masks = [torch.zeros(sum(self.n_p)) for _ in range(self.i_attractor)]
        n_p = np.cumsum(np.concatenate(([0], self.n_p)))

        # For each frequency, insert ones in the mask for those iterations
        for f, max_i in enumerate(self.i_attractor_max_freq_inf):
            for i in range(max_i):
                masks[i][n_p[f] : n_p[f + 1]] = 1.0

        return masks

    @computed_field
    @property
    def p_retrieve_mask_gen(self) -> list[torch.Tensor]:
        """Hierarchical memory retrieval masks for generative model."""
        masks = [torch.zeros(sum(self.n_p)) for _ in range(self.i_attractor)]
        n_p = np.cumsum(np.concatenate(([0], self.n_p)))

        # For each frequency, insert ones in the mask for those iterations
        for f, max_i in enumerate(self.i_attractor_max_freq_gen):
            for i in range(max_i):
                masks[i][n_p[f] : n_p[f + 1]] = 1.0

        return masks

    @computed_field
    @property
    def g_connections(self) -> list[list[bool]]:
        """
        In path integration, abstract location frequency modules can influence the transition of other modules
        hierarchically (low to high).
        """
        # Connections for grid cell modules
        connections = [[self.f_initial[f_from] <= self.f_initial[f_to] for f_from in range(self.n_f_g)] + [False for _ in range(self.n_f_ovc)] for f_to in range(self.n_f_g)]

        # Add connections for separate object vector cell modules
        connections += [
            [False for _ in range(self.n_f_g)] + [self.f_initial[f_from] <= self.f_initial[f_to] for f_from in range(self.n_f_g, self.n_f)] for f_to in range(self.n_f_g, self.n_f)
        ]

        return connections

    # ---- Static matrices
    @computed_field
    @property
    def W_repeat(self) -> list[torch.Tensor]:
        """Matrix for repeating abstract location g to do outer product with sensory information x."""
        return [torch.tensor(np.kron(np.eye(self.n_g_subsampled[f]), np.ones((1, self.n_x_f[f]))), dtype=torch.float) for f in range(self.n_f)]

    @computed_field
    @property
    def W_tile(self) -> list[torch.Tensor]:
        """Matrix for tiling sensory observation x to do outer product with abstract location g."""
        return [torch.tensor(np.kron(np.ones((1, self.n_g_subsampled[f])), np.eye(self.n_x_f[f])), dtype=torch.float) for f in range(self.n_f)]

    @computed_field
    @property
    def two_hot_table(self) -> list[torch.Tensor]:
        """Table for converting one-hot to two-hot compressed representation."""
        table = [[0] * (self.n_x_c - 2) + [1] * 2]

        # Generate compressed codes for each possible observation
        for i in range(1, min(int(comb(self.n_x_c, 2)), self.n_x)):
            code = table[-1].copy()
            # Find latest occurrence of [0 1] in that code
            swap = [index for index in range(len(code) - 1, -1, -1) if code[index : index + 2] == [0, 1]][0]
            # Swap those to get new code
            code[swap : swap + 2] = [1, 0]
            # If the first one was swapped: value after swapped pair is 1
            if swap + 2 < len(code) and code[swap + 2] == 1:
                # Move the second 1 all the way back - reverse everything after the swapped pair
                code[swap + 2 :] = code[: swap + 1 : -1]
            table.append(code)

        # Convert each code to column vector pytorch tensor
        return [torch.tensor(code) for code in table]

    @computed_field
    @property
    def g_downsample(self) -> list[torch.Tensor]:
        """Downsampling matrix to go from grid cells to compressed grid cells."""
        return [
            torch.cat([torch.eye(dim_out, dtype=torch.float), torch.zeros((dim_in - dim_out, dim_out), dtype=torch.float)])
            for dim_in, dim_out in zip(self.n_g, self.n_g_subsampled)
        ]

    model_config = {"populate_by_name": True, "arbitrary_types_allowed": True}


@dataclass
class RuntimeHyperparameters:
    """Runtime hyperparameters injected by training (not part of model architecture).

    These values are computed by the training schedule and updated each step.
    They control time-varying aspects of model behavior during training.
    """

    eta: float = 0.0  # Hebbian learning rate (rate of remembering)
    hebbian_decay: float = 0.9999  # Hebbian decay factor (rate of forgetting)
    p2g_scale_offset: float = 1.0  # Variance offset scaling for p->g inference


# This specifies how parameters are updated at every backpropagation iteration/gradient update
def parameter_iteration(iteration, params):
    # Calculate eta (rate of remembering) and hebian decay (rate of forgetting) for Hebbian memory updates
    hebbian_decay = params.get("hebbian_decay")
    eta = min((iteration + 1) / params["eta_it"], 1) * params["eta"]
    lamb = min((iteration + 1) / params["lambda_it"], 1) * hebbian_decay
    # Calculate current scaling of variance offset for ground location inference
    p2g_scale_offset = 1 / (1 + np.exp((iteration - params["p2g_sig_half_it"]) / params["p2g_sig_scale_it"]))
    # Calculate current learning rate
    lr = max(params["lr_min"] + (params["lr_max"] - params["lr_min"]) * (params["lr_decay_rate"] ** (iteration / params["lr_decay_steps"])), params["lr_min"])
    # Calculate center of walk length window, within which the walk lenghts of new walks are uniformly sampled
    # Use max_steps (Lightning's Trainer.max_steps) as the training horizon
    max_steps = max(int(params.get("max_steps", 1)), 1)
    walk_length_center = (
        params["walk_it_max"] - params["walk_it_window"] * 0.5 - min((iteration + 1) / max_steps, 1) * (params["walk_it_max"] - params["walk_it_min"] - params["walk_it_window"])
    )
    # Calculate current loss weights
    L_p_g = min((iteration + 1) / params["loss_weights_p_g_it"], 1) * params["loss_weights_p"]
    L_p_x = min((iteration + 1) / params["loss_weights_p_g_it"], 1) * params["loss_weights_p"] * (1 - p2g_scale_offset)
    L_x_gen = params["loss_weights_x"]
    L_x_g = params["loss_weights_x"]
    L_x_p = params["loss_weights_x"]
    L_g = min((iteration + 1) / params["loss_weights_p_g_it"], 1) * params["loss_weights_g"]
    L_reg_g = (1 - min((iteration + 1) / params["loss_weights_reg_g_it"], 1)) * params["loss_weights_reg_g"]
    L_reg_p = (1 - min((iteration + 1) / params["loss_weights_reg_p_it"], 1)) * params["loss_weights_reg_p"]
    # And concatenate them in the order expected by the model
    loss_weights = torch.tensor([L_p_g, L_p_x, L_x_gen, L_x_g, L_x_p, L_g, L_reg_g, L_reg_p])
    # Return all updated parameters
    return eta, lamb, p2g_scale_offset, lr, walk_length_center, loss_weights


class TEMModel(torch.nn.Module):
    def __init__(self, params):
        # First call super class init function to set up torch.nn.Module style model and inherit it's functionality
        super(TEMModel, self).__init__()
        # Copy hyperparameters (e.g. network sizes) from parameter dict, usually generated from parameters() in parameters.py
        self.hyper = copy.deepcopy(params)

        # Initialize runtime hyperparameters with safe defaults
        # These will be updated by training before each forward pass
        self.runtime = RuntimeHyperparameters()

        # Create trainable parameters
        self.init_trainable()

    def set_runtime_hyperparams(self, eta: float, hebbian_decay: float, p2g_scale_offset: float) -> None:
        """Set runtime hyperparameters (called by training loop each step).

        Args:
            eta: Hebbian learning rate (rate of remembering)
            hebbian_decay: Hebbian decay factor (rate of forgetting)
            p2g_scale_offset: Variance offset scaling for p->g inference
        """
        self.runtime.eta = eta
        self.runtime.hebbian_decay = hebbian_decay
        self.runtime.p2g_scale_offset = p2g_scale_offset

    def _apply(self, fn):
        """Override _apply to move tensors in self.hyper when model is moved to GPU/CPU."""
        super()._apply(fn)
        self.hyper = self._apply_to_nested_tensors(self.hyper, fn)
        return self

    def _apply_to_nested_tensors(self, obj, fn):
        """Recursively apply function to all tensors in nested dict/list/tuple structure."""
        if torch.is_tensor(obj):
            return fn(obj)
        if isinstance(obj, dict):
            return {k: self._apply_to_nested_tensors(v, fn) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._apply_to_nested_tensors(v, fn) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._apply_to_nested_tensors(v, fn) for v in obj)
        return obj

    def forward(self, x, locations, a_prev, M_prev, x_prev, g_prev):
        # First, do the transition step, as it will be necessary for both the inference and generative part of the model
        gt_gen, gt_inf = self.gen_g(a_prev, g_prev, locations)
        # Run inference model: infer grounded location p_inf (hippocampus), abstract location g_inf (entorhinal). Also keep filtered sensory observation (x_inf), and retrieved grounded location p_inf_x
        x_inf, g_inf, p_inf_x, p_inf = self.inference(x, locations, M_prev, x_prev, gt_inf)
        # Run generative model: since generative model is only used for training purposes, it will generate from *inferred* variables instead of *generated* variables (as it would when used for generation)
        x_gen, x_logits, p_gen = self.generative(M_prev, p_inf, g_inf, gt_gen)
        # Update generative memory with generated and inferred grounded location.
        M = [self.hebbian(M_prev[0], torch.cat(p_inf, dim=1), torch.cat(p_gen, dim=1))]
        # If using memory for grounded location inference: append inference memory
        if self.hyper["use_p_inf"]:
            # Inference memory is identical to generative memory if using common memory, and updated separatedly if not
            M.append(M[0] if self.hyper["common_memory"] else self.hebbian(M_prev[1], torch.cat(p_inf, dim=1), torch.cat(p_inf_x, dim=1), do_hierarchical_connections=False))
        # Return all iteration values (loss now computed in Lightning module)
        return M, gt_gen, p_gen, x_gen, x_logits, x_inf, g_inf, p_inf, p_inf_x

    def inference(self, x, locations, M_prev, x_prev, g_gen):
        # Compress sensory observation from one-hot to two-hot (or alternatively, whatever an MLP makes of it)
        x_c = self.f_c(x)
        # Temporally filter sensory observation by mixing it with previous experience
        x_f = self.x_prev2x(x_prev, x_c)
        # Prepare sensory experience for input to memory by normalisation and weighting
        x_ = self.x2x_(x_f)
        # Retrieve grounded location from memory by doing pattern completion on current sensory experience
        p_x = self.attractor(x_, M_prev[1], retrieve_it_mask=self.hyper["p_retrieve_mask_inf"]) if self.hyper["use_p_inf"] else None
        # Infer abstract location by combining previous abstract location and grounded location retrieved from memory by current sensory experience
        g = self.inf_g(p_x, g_gen, x, locations)
        # Prepare abstract location for input to memory by downsampling and weighting
        g_ = self.g2g_(g)
        # Infer grounded location from sensory experience and inferred abstract location
        p = self.inf_p(x_, g_)
        # Return variables in order that they were created
        return x_f, g, p_x, p

    def generative(self, M_prev, p_inf, g_inf, g_gen):
        # Generate observation from inferred grounded location, using only the highest frequency. Also keep non-softmaxed logits which are used in the loss later
        x_p, x_p_logits = self.gen_x(p_inf[0])
        # Retrieve grounded location from memory by pattern completion on inferred abstract location
        p_g_inf = self.gen_p(g_inf, M_prev[0])  # was p_mem_gen
        # And generate observation from the grounded location retrieved from inferred abstract location
        x_g, x_g_logits = self.gen_x(p_g_inf[0])
        # Retreive grounded location from memory by pattern completion on abstract location by transitioning
        p_g_gen = self.gen_p(g_gen, M_prev[0])
        # Generate observation from sampled grounded location
        x_gt, x_gt_logits = self.gen_x(p_g_gen[0])
        # Return all generated observations and their corresponding logits
        return (x_p, x_g, x_gt), (x_p_logits, x_g_logits, x_gt_logits), p_g_inf

    def init_trainable(self):
        # Scale factor in Laplacian transform for each frequency module. High frequency comes first, low frequency comes last. Learn inverse sigmoid instead of scale factor directly, so domain of alpha is -inf, inf
        self.alpha = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(np.log(self.hyper["f_initial"][f] / (1 - self.hyper["f_initial"][f])), dtype=torch.float)) for f in range(self.hyper["n_f"])]
        )
        # Entorhinal preference weights
        self.w_x = torch.nn.Parameter(torch.tensor(1.0))
        # Entorhinal preference bias
        self.b_x = torch.nn.Parameter(torch.zeros(self.hyper["n_x_c"]))
        # Frequency module specific scaling of sensory experience before input to hippocampus
        self.w_p = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(1.0)) for f in range(self.hyper["n_f"])])
        # Initial activity of abstract location cells when entering a new environment, like a prior on g. Initialise with truncated normal
        self.g_init = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.tensor(truncnorm.rvs(-2, 2, size=self.hyper["n_g"][f], loc=0, scale=self.hyper["g_init_std"]), dtype=torch.float))
                for f in range(self.hyper["n_f"])
            ]
        )
        # Log of standard deviation of abstract location cells when entering a new environment; standard deviation of the prior on g. Initialise with truncated normal
        self.logsig_g_init = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.tensor(truncnorm.rvs(-2, 2, size=self.hyper["n_g"][f], loc=0, scale=self.hyper["g_init_std"]), dtype=torch.float))
                for f in range(self.hyper["n_f"])
            ]
        )
        # MLP for transition weights (not in paper, but recommended by James so you can learn about similarities between actions). Size is given by grid connections
        self.MLP_D_a = MLP(
            [self.hyper["n_actions"] for _ in range(self.hyper["n_f"])],
            [
                sum([self.hyper["n_g"][f_from] for f_from in range(self.hyper["n_f"]) if self.hyper["g_connections"][f_to][f_from]]) * self.hyper["n_g"][f_to]
                for f_to in range(self.hyper["n_f"])
            ],
            activation=[torch.tanh, None],
            hidden_dim=[self.hyper["d_hidden_dim"] for _ in range(self.hyper["n_f"])],
            bias=[True, False],
        )
        # Initialise the hidden to output weights as zero, so initially you simply keep the current abstract location to predict the next abstract location
        self.MLP_D_a.set_weights(1, 0.0)
        # Transition weights without specifying an action for use in generative model with shiny objects
        self.D_no_a = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.zeros(sum([self.hyper["n_g"][f_from] for f_from in range(self.hyper["n_f"]) if self.hyper["g_connections"][f_to][f_from]]) * self.hyper["n_g"][f_to])
                )
                for f_to in range(self.hyper["n_f"])
            ]
        )
        # MLP for standard deviation of transition sample
        self.MLP_sigma_g_path = MLP(self.hyper["n_g"], self.hyper["n_g"], activation=[torch.tanh, torch.exp], hidden_dim=[2 * g for g in self.hyper["n_g"]])
        # MLP for standard devation of grounded location from retrieved memory sample
        self.MLP_sigma_p = MLP(self.hyper["n_p"], self.hyper["n_p"], activation=[torch.tanh, torch.exp])
        # MLP to generate mean of abstract location from downsampled abstract location, obtained by summing grounded location over sensory preferences in inference model
        self.MLP_mu_g_mem = MLP(self.hyper["n_g_subsampled"], self.hyper["n_g"], hidden_dim=[2 * g for g in self.hyper["n_g"]])
        # Initialise weights in last layer of MLP_mu_g_mem as truncated normal for each frequency module
        self.MLP_mu_g_mem.set_weights(
            -1,
            [
                torch.tensor(truncnorm.rvs(-2, 2, size=list(self.MLP_mu_g_mem.w[f][-1].weight.shape), loc=0, scale=self.hyper["g_mem_std"]), dtype=torch.float)
                for f in range(self.hyper["n_f"])
            ],
        )
        # MLP to generate standard deviation of abstract location from two measures (generated observation error and inferred abstract location vector norm) of memory quality
        self.MLP_sigma_g_mem = MLP([2 for _ in self.hyper["n_g_subsampled"]], self.hyper["n_g"], activation=[torch.tanh, torch.exp], hidden_dim=[2 * g for g in self.hyper["n_g"]])
        # MLP to generate mean of abstract location directly from shiny object presence. Outputs to object vector cell modules if they're separated, else to all abstract location modules
        self.MLP_mu_g_shiny = MLP(
            [1 for _ in range(self.hyper["n_f_ovc"] if self.hyper["separate_ovc"] else self.hyper["n_f"])],
            [n_g for n_g in self.hyper["n_g"][(self.hyper["n_f_g"] if self.hyper["separate_ovc"] else 0) :]],
            hidden_dim=[2 * n_g for n_g in self.hyper["n_g"][(self.hyper["n_f_g"] if self.hyper["separate_ovc"] else 0) :]],
        )
        # MLP to generate standard deviation of abstract location directly from shiny object presence. Outputs to object vector cell modules if they're separated, else to all abstract location modules
        self.MLP_sigma_g_shiny = MLP(
            [1 for _ in range(self.hyper["n_f_ovc"] if self.hyper["separate_ovc"] else self.hyper["n_f"])],
            [n_g for n_g in self.hyper["n_g"][(self.hyper["n_f_g"] if self.hyper["separate_ovc"] else 0) :]],
            hidden_dim=[2 * n_g for n_g in self.hyper["n_g"][(self.hyper["n_f_g"] if self.hyper["separate_ovc"] else 0) :]],
            activation=[torch.tanh, torch.exp],
        )
        # MLP for decompressing highest frequency sensory experience to sensory observation
        self.MLP_c_star = MLP(self.hyper["n_x_f"][0], self.hyper["n_x"], hidden_dim=20 * self.hyper["n_x_c"])

    def init_iteration(self, g, x, a, M):
        # On the very first iteration, update the batch size based on the data. This is useful when doing analysis on the network with different batch sizes compared to training
        self.hyper["batch_size"] = x.shape[0]
        # Initalise hebbian memory connectivity matrix [M_gen, M_inf] if it wasn't initialised yet
        if M is None:
            # Create new empty memory dict for generative network: zero connectivity matrix M_0, then empty list of the memory vectors a and b for each iteration for efficient hebbian memory computation
            M = [torch.zeros((self.hyper["batch_size"], sum(self.hyper["n_p"]), sum(self.hyper["n_p"])), dtype=torch.float, device=x.device)]
            # Append inference memory only if memory is used in grounded location inference
            if self.hyper["use_p_inf"]:
                # If inference and generative network share common memory: reuse same connectivity, and same memory vectors. Else, create a new empty memory list for inference network
                M.append(
                    M[0]
                    if self.hyper["common_memory"]
                    else torch.zeros((self.hyper["batch_size"], sum(self.hyper["n_p"]), sum(self.hyper["n_p"])), dtype=torch.float, device=x.device)
                )
        # Initialise previous abstract location by stacking abstract location prior
        g_inf = [torch.stack([self.g_init[f] for _ in range(self.hyper["batch_size"])]) for f in range(self.hyper["n_f"])]
        # Initialise previous sensory experience with zeros, as there is no data yet for temporal smoothing
        x_inf = [torch.zeros((self.hyper["batch_size"], self.hyper["n_x_f"][f]), device=x.device) for f in range(self.hyper["n_f"])]
        # And construct new iteration for that g, x, a, and M
        return TEMState(g=g, x=x, a=a, M=M, x_inf=x_inf, g_inf=g_inf)

    def gen_g(self, a_prev, g_prev, locations):
        # Transition from previous abstract location to new abstract location using weights specific to action taken for each frequency module
        mu_g = self.f_mu_g_path(a_prev, g_prev)
        sigma_g = self.f_sigma_g_path(a_prev, g_prev)
        # Either sample new abstract location g or simply take the mean of distribution in noiseless case.
        g = [mu_g[f] + sigma_g[f] * np.random.randn() if self.hyper["do_sample"] else mu_g[f] for f in range(self.hyper["n_f"])]
        # But for environments with shiny objects, the transition to the new abstract location shouldn't have access to the action direction in the generative model
        shiny_envs = [location["shiny"] is not None for location in locations]
        # If there are any shiny environments, the abstract locations for the generative model will need to be re-calculated without providing actions for those
        g_gen = self.f_mu_g_path(a_prev, g_prev, no_direc=shiny_envs) if any(shiny_envs) else g
        # Return generated abstract location after transition
        return g_gen, (g, sigma_g)

    def gen_p(self, g, M_prev):
        # We want to use g as an index for memory retrieval, but it doesn't have the right dimensions (these are grid cells, we need place cells). We need g_ instead
        g_ = self.g2g_(g)
        # Retreive memory: do pattern completion on abstract location to get grounded location
        mu_p = self.attractor(g_, M_prev, retrieve_it_mask=self.hyper["p_retrieve_mask_gen"])
        sigma_p = self.f_sigma_p(mu_p)
        # Either sample new grounded location p or simply take the mean of distribution in noiseless case
        p = [mu_p[f] + sigma_p[f] * np.random.randn() if self.hyper["do_sample"] else mu_p[f] for f in range(self.hyper["n_f"])]
        # Return pattern-completed grounded location p after memory retrieval
        return p

    def gen_x(self, p):
        # Get categorical distribution over observations from grounded location
        # If you actually want to sample observation, you need a reparaterisation trick for categorical distributions
        # Sampling would be the correct way to do this, since observations are discrete, and it's also what the TEM paper says
        # However, it looks like you could also get away with using categorical distribution directly as an approximation of the one-hot observations
        if self.hyper["do_sample"]:
            x, logits = self.f_x(
                p
            )  # This is a placeholder! Should be done using reparameterisation trick (like https://blog.evjang.com/2016/11/tutorial-categorical-variational.html)
        else:
            x, logits = self.f_x(p)
        # Return one-hot (or almost one-hot...) observation obtained from grounded location, and also the non-softmaxed logits
        return x, logits

    def inf_g(self, p_x, g_gen, x, locations):
        # Infer abstract location from the combination of [grounded location retrieved from memory by sensory experience] ...
        if self.hyper["use_p_inf"]:
            # Not in paper, but makes sense from symmetry with f_x: first get g from p by "summing over sensory preferences" g = p * W_repeat^T
            g_downsampled = [torch.matmul(p_x[f], torch.t(self.hyper["W_repeat"][f])) for f in range(self.hyper["n_f"])]
            # Then use abstract location after summing over sensory preferences as input to MLP to obtain the inferred abstract location from memory
            mu_g_mem = self.f_mu_g_mem(g_downsampled)
            # Not in paper, but this greatly improves zero-shot inference: provide the uncertainty function of the inferred abstract location with measures of memory quality
            with torch.no_grad():
                # For the first measure, use the grounded location inferred from memory to generate an observation
                x_hat, x_hat_logits = self.gen_x(p_x[0])
                # Then calculate the error between the generated observation and the actual observation: if the memory is working well, this error should be small
                err = utils.squared_error(x, x_hat)
            # The second measure is the vector norm of the inferred abstract location; good memories should have similar vector norms. Concatenate the two measures as input for the abstract location uncertainty function
            sigma_g_input = [torch.cat((torch.sum(g**2, dim=1, keepdim=True), torch.unsqueeze(err, dim=1)), dim=1) for g in mu_g_mem]
            # Not in paper, but recommended by James for stability: get final mean of inferred abstract location by clamping activations between -1 and 1
            mu_g_mem = self.f_g_clamp(mu_g_mem)
            # And get standard deviation/uncertainty of inferred abstract location by providing uncertainty function with memory quality measures
            sigma_g_mem = self.f_sigma_g_mem(sigma_g_input)
        # ... and [previous abstract location and action (path integration)]
        mu_g_path = g_gen[0]
        sigma_g_path = g_gen[1]
        # Infer abstract location by combining previous abstract location and grounded location retrieved from memory by current sensory experience
        mu_g, sigma_g = [], []
        for f in range(self.hyper["n_f"]):
            if self.hyper["use_p_inf"]:
                # Then get full gaussian distribution of inferred abstract location by calculating precision weighted mean
                mu, sigma = utils.inv_var_weight([mu_g_path[f], mu_g_mem[f]], [sigma_g_path[f], sigma_g_mem[f]])
            else:
                # Or simply completely ignore the inference memory here, to test if things are working
                mu, sigma = mu_g_path[f], sigma_g_path[f]
            # Append mu and sigma to list for all frequency modules
            mu_g.append(mu)
            sigma_g.append(sigma)
        # Finally (though not in paper), also add object vector cell information to inferred abstract location for environments with shiny objects
        shiny_envs = [location["shiny"] is not None for location in locations]
        if any(shiny_envs):
            # Find for which environments the current location has a shiny object
            shiny_locations = torch.unsqueeze(torch.stack([torch.tensor(location["shiny"], dtype=torch.float) for location in locations if location["shiny"] is not None]), dim=-1)
            # Get abstract location for environments with shiny objects and feed to each of the object vector cell modules
            mu_g_shiny = self.f_mu_g_shiny([shiny_locations for _ in range(self.hyper["n_f_g"] if self.hyper["separate_ovc"] else self.hyper["n_f"])])
            sigma_g_shiny = self.f_sigma_g_shiny([shiny_locations for _ in range(self.hyper["n_f_g"] if self.hyper["separate_ovc"] else self.hyper["n_f"])])
            # Update only object vector modules with shiny-inferred abstract location: start from offset if object vector modules are separate
            module_start = self.hyper["n_f_g"] if self.hyper["separate_ovc"] else 0
            # Inverse variance weighting is associative, so I can just do additional inverse variance weighting to the previously obtained mu and sigma - but only for object vector cell modules!
            for f in range(module_start, self.hyper["n_f"]):
                # Add inferred abstract location from shiny objects to previously obtained position, only for environments with shiny objects
                mu, sigma = utils.inv_var_weight([mu_g[f][shiny_envs, :], mu_g_shiny[f - module_start]], [sigma_g[f][shiny_envs, :], sigma_g_shiny[f - module_start]])
                # In order to update only the environments with shiny objects, without in-place value assignment, construct a mask of shiny environments
                mask = torch.zeros_like(mu_g[f], dtype=torch.bool)
                mask[shiny_envs, :] = True
                # Use mask to update the shiny environment entries in inferred abstract locations
                mu_g[f] = mu_g[f].masked_scatter(mask, mu)
                sigma_g[f] = sigma_g[f].masked_scatter(mask, sigma)
        # Either sample inferred abstract location from combined (precision weighted) distribution or just take mean
        g = [mu_g[f] + sigma_g[f] * np.random.randn() if self.hyper["do_sample"] else mu_g[f] for f in range(self.hyper["n_f"])]
        # Return abstract location inferred from grounded location from memory and previous abstract location
        return g

    def inf_p(self, x_, g_):
        # Infer grounded location from sensory experience and inferred abstract location for each module
        p = []
        # Use the same transformation for each frequency module: leaky relu for sparsity
        for f in range(self.hyper["n_f"]):
            mu_p = self.f_p(g_[f] * x_[f])  # This is element-wise multiplication
            sigma_p = 0  # Unclear from paper (typo?). Some undefined function f that takes two arguments: f(f_n(x),g)
            # Either sample inferred grounded location or just take mean
            if self.hyper["do_sample"]:
                p.append(mu_p + sigma_p * np.random.randn())
            else:
                p.append(mu_p)
        # Return new memory constructed from sensory experience and inferred abstract location
        return p

    def x_prev2x(self, x_prev, x_c):
        # Calculate factor for filtering from sigmoid of learned parameter
        alpha = [torch.nn.Sigmoid()(self.alpha[f]) for f in range(self.hyper["n_f"])]
        # Do exponential temporal filtering for each frequency modulemod
        x = [(1 - alpha[f]) * x_prev[f] + alpha[f] * x_c for f in range(self.hyper["n_f"])]
        return x

    def x2x_(self, x):
        # Prepare sensory input for input to memory by weighting and normalisation for each frequency module
        # Get normalised sensory input for each frequency module
        normalised = self.f_n(x)
        # Then reshape and reweight (use sigmoid to keep weight between 0 and 1) each frequency module separately: matrix multiplication by W_tile prepares x for outer product with g by element-wise multiplication
        x_ = [torch.nn.Sigmoid()(self.w_p[f]) * torch.matmul(normalised[f], self.hyper["W_tile"][f]) for f in range(self.hyper["n_f"])]
        return x_

    def g2g_(self, g):
        # Prepares abstract location for input to memory by reshaping and down-sampling for each frequency module
        # Get downsampled abstract location for each frequency module
        downsampled = self.f_g(g)
        # Then reshape and reweight each frequency module separately
        g_ = [torch.matmul(downsampled[f], self.hyper["W_repeat"][f]) for f in range(self.hyper["n_f"])]
        return g_

    def f_mu_g_path(self, a_prev, g_prev, no_direc=None):
        # If there are no environments where the transition direction needs to be omitted (e.g. no shiny objects, or in inference model: set to all false
        no_direc = [False for _ in a_prev] if no_direc is None else no_direc
        # Remove all Nones from a_prev: these are walks where there was no previous action, so no step needs to be calculated for those
        a_prev_step = [a if a is not None else 0 for a in a_prev]
        # And also keep track of which walks these valid step actions are for
        a_do_step = [a != None for a in a_prev]
        # Get device from g_prev
        device = g_prev[0].device
        # Transform list of actions into batch of one-hot row vectors.
        if self.hyper["has_static_action"]:
            # If this world has static actions: whenever action 0 (standing still) appears, the action vector should be all zeros. All other actions should have a 1 in the label-1 entry
            a = torch.zeros((len(a_prev_step), self.hyper["n_actions"]), device=device).scatter_(
                1, torch.clamp(torch.tensor(a_prev_step, device=device).unsqueeze(1) - 1, min=0), 1.0 * (torch.tensor(a_prev_step, device=device).unsqueeze(1) > 0)
            )
        else:
            # Without static actions: each action label should become a one-hot vector for that label
            a = torch.zeros((len(a_prev_step), self.hyper["n_actions"]), device=device).scatter_(1, torch.tensor(a_prev_step, device=device).unsqueeze(1), 1.0)
        # Get vector of transition weights by feeding actions into MLP
        D_a = self.MLP_D_a([a for _ in range(self.hyper["n_f"])])
        # Replace transition weights by non-directional transition weights in environments where transition direction needs to be omitted (can set only if any no_direc)
        for f in range(self.hyper["n_f"]):
            D_a[f][no_direc, :] = self.D_no_a[f]
        # Reshape transition weight vector into transition matrix. The number of rows in the transition matrix is given by the incoming abstract location connections for each frequency module
        D_a = [
            torch.reshape(
                D_a[f_to], (-1, sum([self.hyper["n_g"][f_from] for f_from in range(self.hyper["n_f"]) if self.hyper["g_connections"][f_to][f_from]]), self.hyper["n_g"][f_to])
            )
            for f_to in range(self.hyper["n_f"])
        ]
        # Select the frequency modules of the previous abstract location that are connected to each frequency module, to
        g_in = [
            torch.unsqueeze(torch.cat([g_prev[f_from] for f_from in range(self.hyper["n_f"]) if self.hyper["g_connections"][f_to][f_from]], dim=1), 1)
            for f_to in range(self.hyper["n_f"])
        ]
        # Reshape transition weight vector into transition matrix. The number of rows in the transition matrix is given by the incoming abstract location connections for each frequency module
        delta = [torch.squeeze(torch.matmul(g, T)) for g, T in zip(g_in, D_a)]
        # Not in the paper, but recommended by James for stability: use inferred code as *difference* in abstract location. Calculate new abstract location from previous abstract location and difference
        g_step = [g + d if g.dim() > 1 else torch.unsqueeze(g + d, 0) for g, d in zip(g_prev, delta)]
        # Not in paper, but recommended by James for stability: clamp activations between -1 and 1
        g_step = self.f_g_clamp(g_step)
        # Build new abstract location from result of transition if there was one, or from prior on abstract location if there wasn't
        return [torch.stack([g_step[f][batch_i, :] if do_step else self.g_init[f] for batch_i, do_step in enumerate(a_do_step)]) for f in range(self.hyper["n_f"])]

    def f_sigma_g_path(self, a_prev, g_prev):
        # Keep track of which walks these valid step actions are for
        a_do_step = [a != None for a in a_prev]
        # Multi layer perceptron to generate standard deviation from all previous abstract locations, including those that were just initialised and not real previous locations
        from_g = self.MLP_sigma_g_path(g_prev)
        # And take exponent to get prior sigma for the walks that didn't have a previous location
        from_prior = [torch.exp(logsig) for logsig in self.logsig_g_init]
        # Now select the standard deviation generated from the previous abstract location if there was one, and the prior standard deviation on abstract location otherwise
        return [torch.stack([from_g[f][batch_i, :] if do_step else from_prior[f] for batch_i, do_step in enumerate(a_do_step)]) for f in range(self.hyper["n_f"])]

    def f_mu_g_mem(self, g_downsampled):
        # Multi layer perceptron to generate mean of abstract location from down-sampled abstract location, obtained by summing over sensory dimension of grounded location
        return self.MLP_mu_g_mem(g_downsampled)

    def f_sigma_g_mem(self, g_downsampled):
        # Multi layer perceptron to generate standard deviation of abstract location from down-sampled abstract location, obtained by summing over sensory dimension of grounded location
        sigma = self.MLP_sigma_g_mem(g_downsampled)
        # Not in paper, but also offset this sigma over training, so you can reduce influence of inferred p early on (from runtime, not hyper)
        return [sigma[f] + self.runtime.p2g_scale_offset * self.hyper["p2g_sig_val"] for f in range(self.hyper["n_f"])]

    def f_mu_g_shiny(self, shiny):
        # Multi layer perceptron to generate mean of abstract location from boolean location shiny-ness
        mu_g = self.MLP_mu_g_shiny(shiny)
        # Take absolute because James wants object vector cells to be positive
        mu_g = [torch.abs(mu) for mu in mu_g]
        # Then apply clamp and leaky relu to get object vector module activations, like it's done for ground location activations
        g = self.f_p(mu_g)
        return g

    def f_sigma_g_shiny(self, shiny):
        # Multi layer perceptron to generate standard deviation of abstract location from boolean location shiny-ness
        return self.MLP_sigma_g_shiny(shiny)

    def f_sigma_p(self, p):
        # Multi layer perceptron to generate standard deviation of grounded location retrieval
        return self.MLP_sigma_p(p)

    def f_x(self, p):
        # Calculate categorical probability distribution over observations for a given ground location
        # p has dimensions n_p[0]. We'll need to transform those to temporally filtered sensory experience, before we can decompress
        # p is the flattened (by concatenating rows - like reading sentences) outer product of g and x (p = g^T * x).
        # Therefore to get the sensory experience x for a grounded location p, sum over all abstract locations g for each component of x
        # That's what the paper means when it says "sum over entorhinal preferences". It can be done with the transpose of W_tile
        x = self.w_x * torch.matmul(p, torch.t(self.hyper["W_tile"][0])) + self.b_x
        # Then we need to decompress the temporally filtered sensory experience into a single current experience prediction
        logits = self.f_c_star(x)
        # We'll keep both the logits (domain -inf, inf) and probabilities (domain 0, 1) because both are needed later on
        probability = utils.softmax(logits)
        return probability, logits

    def f_c_star(self, compressed):
        # Multi layer perceptron to decompress sensory experience at highest frequency
        return self.MLP_c_star(compressed)

    def f_c(self, decompressed):
        # Compress sensory observation from one-hot provided by world to two-hot for ease of computation
        return torch.stack([self.hyper["two_hot_table"][i] for i in torch.argmax(decompressed, dim=1)], dim=0)

    def f_n(self, x):
        # Normalise sensory observation for each frequency module
        normalised = [utils.normalise(utils.relu(x[f] - torch.mean(x[f]))) for f in range(self.hyper["n_f"])]
        return normalised

    def f_g(self, g):
        # Downsample abstract location for each frequency module
        downsampled = [torch.matmul(g[f], self.hyper["g_downsample"][f]) for f in range(self.hyper["n_f"])]
        return downsampled

    def f_g_clamp(self, g):
        # Calculate activation for abstract location, thresholding between -1 and 1
        activation = [torch.clamp(g_f, min=-1, max=1) for g_f in g]
        return activation

    def f_p(self, p):
        # Calculate activation for inferred grounded location, using a leaky relu for sparsity. Either apply to full multi-frequency grounded location or single frequency module
        activation = [utils.leaky_relu(torch.clamp(p_f, min=-1, max=1)) for p_f in p] if type(p) is list else utils.leaky_relu(torch.clamp(p, min=-1, max=1))
        return activation

    def attractor(self, p_query, M, retrieve_it_mask=None):
        # Retreive grounded location from attractor network memory with weights M by pattern-completing query
        # For example, initial attractor input can come from abstract location (g_) or sensory experience (x_)
        # Start by flattening query grounded locations across frequency modules
        h_t = torch.cat(p_query, dim=1)
        # Apply activation function to initial memory index
        h_t = self.f_p(h_t)
        # Hierarchical retrieval (not in paper) is implemented by early stopping retrieval for low frequencies, using a mask. If not specified: initialise mask as all 1s
        retrieve_it_mask = [torch.ones(sum(self.hyper["n_p"])) for _ in range(self.hyper["n_p"])] if retrieve_it_mask is None else retrieve_it_mask
        # Iterate attractor dynamics to do pattern completion
        for tau in range(self.hyper["i_attractor"]):
            # Apply one iteration of attractor dynamics, but only where there is a 1 in the mask. NB retrieve_it_mask entries have only one row, but are broadcasted to batch_size
            h_t = (1 - retrieve_it_mask[tau]) * h_t + retrieve_it_mask[tau] * (self.f_p(self.hyper["kappa"] * h_t + torch.squeeze(torch.matmul(torch.unsqueeze(h_t, 1), M))))
        # Make helper list of cumulative neurons per frequency module for grounded locations
        n_p = np.cumsum(np.concatenate(([0], self.hyper["n_p"])))
        # Now re-cast the grounded location into different frequency modules, since memory retrieval turned it into one long vector
        p = [h_t[:, n_p[f] : n_p[f + 1]] for f in range(self.hyper["n_f"])]
        return p

    def hebbian(self, M_prev, p_inferred, p_generated, do_hierarchical_connections=True):
        # Create new ground memory for attractor network by setting weights to outer product of learned vectors
        # p_inferred corresponds to p in the paper, and p_generated corresponds to p^.
        # The order of p + p^ and p - p^ is reversed since these are row vectors, instead of column vectors in the paper.
        M_new = torch.squeeze(torch.matmul(torch.unsqueeze(p_inferred + p_generated, 2), torch.unsqueeze(p_inferred - p_generated, 1)))
        # Multiply by connection vector, e.g. only keeping weights from low to high frequencies for hierarchical retrieval
        if do_hierarchical_connections:
            M_new = M_new * self.hyper["p_update_mask"]
        # Store grounded location in attractor network memory with weights M by Hebbian learning of pattern
        # Rate of remembering controlled by eta, rate of forgetting by hebbian_decay (from runtime, not hyper)
        M = torch.clamp(self.runtime.hebbian_decay * M_prev + self.runtime.eta * M_new, min=-1, max=1)
        return M


@dataclass
class TEMState:
    """Outputs from a single timestep TEM iteration."""

    g: Any = None
    x: Optional[Tensor] = None
    a: Any = None

    M: Optional[List[Tensor]] = None

    g_gen: Optional[List[Tensor]] = None
    p_gen: Optional[List[Tensor]] = None

    x_gen: Optional[Sequence[Tensor]] = None
    x_logits: Optional[Sequence[Tensor]] = None

    x_inf: Optional[List[Tensor]] = None
    g_inf: Optional[List[Tensor]] = None
    p_inf: Optional[List[Tensor]] = None
    p_inf_x: Optional[List[Tensor]] = None  # Grounded location from sensory input (for loss computation)

    def correct(self) -> List[np.ndarray]:
        """Return per-prediction correctness arrays for the current timestep."""
        if self.x is None or self.x_gen is None:
            return []

        observation = self.x.detach().cpu().numpy()
        predictions = [tensor.detach().cpu().numpy() for tensor in self.x_gen]
        return [np.argmax(pred, axis=-1) == np.argmax(observation, axis=-1) for pred in predictions]

    def detach(self) -> "TEMState":
        """Return a detached copy suitable for storing as `prev_iter`."""

        def _detach(obj: Any) -> Any:
            if obj is None:
                return None
            if torch.is_tensor(obj):
                return obj.detach()
            if isinstance(obj, list):
                return [_detach(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_detach(v) for v in obj)
            if isinstance(obj, dict):
                return {k: _detach(v) for k, v in obj.items()}
            return obj

        return TEMState(
            g=self.g,
            x=_detach(self.x),
            a=self.a,
            M=_detach(self.M),
            g_gen=_detach(self.g_gen),
            p_gen=_detach(self.p_gen),
            x_gen=_detach(self.x_gen),
            x_logits=_detach(self.x_logits),
            x_inf=_detach(self.x_inf),
            g_inf=_detach(self.g_inf),
            p_inf=_detach(self.p_inf),
            p_inf_x=_detach(self.p_inf_x),
        )


Walk = Iterable[Tuple[Any, Tensor, Any]]  # (locations, x, a)


class Rollout(Iterator[TEMState]):
    def __init__(self, model: TEMModel, walk: Walk, initial: Optional[TEMState] = None):
        self.model = model
        self.walk = list(walk)  # Materialize for predictable indexing

        if len(self.walk) == 0:
            raise ValueError("Rollout requires at least 1 timestep in walk")

        # Extract first step to determine batch size and initialize state
        locations_0, x_0, a_0 = self.walk[0]

        # Determine initial state
        if initial is not None:
            prev_state = initial
        else:
            # Create fresh initial state (init_iteration will create memory)
            try:
                batch_size = len(a_0)
            except TypeError:
                # a_0 might not have len() if it's a scalar or tensor
                batch_size = int(x_0.shape[0]) if x_0.ndim > 1 else 1

            prev_state = model.init_iteration(locations_0, x_0, [None for _ in range(batch_size)], None)

        # Initialize prev-values for first forward pass
        self._a_prev = prev_state.a
        self._M_prev = prev_state.M
        self._x_prev = prev_state.x_inf
        self._g_prev = prev_state.g_inf

        # Track current position in walk
        self._idx = 0

    def __iter__(self) -> "Rollout":
        """Return self as iterator."""
        return self

    def __next__(self) -> TEMState:
        """Process next timestep and return state.

        Returns:
            TEMState for current timestep.

        Raises:
            StopIteration: When all timesteps have been processed.
        """
        if self._idx >= len(self.walk):
            raise StopIteration

        # Get current timestep
        locations, x, a = self.walk[self._idx]
        self._idx += 1

        # Run model forward
        M, g_gen, p_gen, x_gen, x_logits, x_inf, g_inf, p_inf, p_inf_x = self.model(x, locations, self._a_prev, self._M_prev, self._x_prev, self._g_prev)

        # Build state
        state = TEMState(
            g=locations,
            x=x,
            a=a,
            M=M,
            g_gen=g_gen,
            p_gen=p_gen,
            x_gen=x_gen,
            x_logits=x_logits,
            x_inf=x_inf,
            g_inf=g_inf,
            p_inf=p_inf,
            p_inf_x=p_inf_x,
        )

        # Update prev-values for next iteration
        self._a_prev = a
        self._M_prev = M
        self._x_prev = x_inf
        self._g_prev = g_inf

        return state
