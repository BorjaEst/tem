"""
TEM training entrypoint.

This module is intentionally "thin": it wires together
- Run settings (Pydantic settings/CLI),
- Run directory layout,
- Model creation/loading,
- And the original training loop.

The goal is to keep behavior stable while making configuration and paths
consistent and easy to maintain.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, NamedTuple, Optional

import numpy as np
import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch.utils.tensorboard import SummaryWriter

from torch_tem import data, utils
from torch_tem.core import model
from torch_tem.core.model import Parameters


def _require_exists(path: Path, what: str) -> None:
    """Raise friendly error if path doesn't exist."""
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")


class RunPaths(NamedTuple):
    """Resolved output paths for a single run."""

    run: Path
    train: Path
    model: Path
    save: Path
    script: Path
    envs: Path


class RunSettings(BaseSettings):
    """Settings for a training run (CLI/env driven)."""

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="run")

    # Load/continue
    load_model: Optional[Path] = Field(default=None, description="Path to existing run directory OR a checkpoint file (tem_*.pt / params_*.pt).")
    i_start: int = Field(default=0, description="Iteration to load when --load-model is a run directory (or to override parsing).")
    override: dict[str, Any] = Field(default_factory=dict, description='Patch loaded params (e.g. {"train_it": 40000}).')

    # Model params (for new runs)
    model_params: Parameters = Field(default_factory=Parameters, description="Model parameters (Pydantic TEM Parameters).")

    # Environments
    envs: list[Path] = Field(default_factory=lambda: [Path("./envs/5x5.json")], description="Environment JSON files for training (only used for new runs).")
    randomise_observations: bool = Field(default=True, description="Randomise observations in environments.")

    # Runtime
    seed: int = Field(default=0, description="Random seed.")
    output_dir: Optional[Path] = Field(default=None, description="Run directory to write into; if omitted uses utils.make_directories() layout.")
    log_every: int = Field(default=10, description="Log every N iterations.")
    save_every: int = Field(default=1000, description="Save checkpoints every N iterations.")


def _resolve_run_paths_for_new_run(settings: RunSettings) -> RunPaths:
    """
    Create (or reuse) run folders for a new training run.

    Returns:
        RunPaths: resolved paths for run/train/model/save/script/envs.
    """
    if settings.output_dir is None:
        # Legacy helper returns strings; convert to Paths.
        run_s, train_s, model_s, save_s, script_s, envs_s = utils.make_directories()
        return RunPaths(
            run=Path(run_s),
            train=Path(train_s),
            model=Path(model_s),
            save=Path(save_s),
            script=Path(script_s),
            envs=Path(envs_s),
        )

    run_path = settings.output_dir
    train_path = run_path / "train"
    model_path = run_path / "model"
    save_path = run_path / "save"
    script_path = run_path / "script"
    envs_path = script_path / "envs"

    for p in (train_path, model_path, save_path, script_path, envs_path):
        p.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        run=run_path,
        train=train_path,
        model=model_path,
        save=save_path,
        script=script_path,
        envs=envs_path,
    )


def params_from_settings(settings: RunSettings) -> dict[str, Any]:
    """
    Create a TEM params dict from the Pydantic Parameters model.

    Returns:
        dict[str, Any]: params dict expected by TEM code.
    """
    params = settings.model_params.model_dump()

    # Backward-compat: some code expects params["lambda"].
    if "lambda" not in params and "lambda_param" in params:
        params["lambda"] = params.pop("lambda_param")

    # Defensive: remove any internal-only fields if present.
    for key in ("n_g_subsampled_base", "n_ovc_base", "f_initial_base"):
        params.pop(key, None)

    return params


def load_existing(settings: RunSettings):
    """
    Load an existing model and its run layout.

    Supports:
      - run directory (contains model/ and script/envs/),
      - checkpoint file tem_*.pt or params_*.pt inside model/.

    Returns:
        (tem, envs, i_start, paths, params)
    """
    if settings.load_model is None:
        raise ValueError("settings.load_model is None")

    load_path = settings.load_model.expanduser().resolve()

    # Determine run/model directories and which iteration to load.
    if load_path.is_file():
        model_dir = load_path.parent
        run_dir = model_dir.parent
        parsed = utils.parse_iter_from_stem(load_path.stem)
        i_load = settings.i_start if settings.i_start else (parsed or 0)

        tem_ckpt = model_dir / f"tem_{i_load}.pt"
        params_ckpt = model_dir / f"params_{i_load}.pt"

        # If user pointed directly at one of the two, respect it.
        if load_path.name.startswith("tem_"):
            tem_ckpt = load_path
        elif load_path.name.startswith("params_"):
            params_ckpt = load_path
    else:
        run_dir = load_path
        model_dir = run_dir / "model"
        i_load = settings.i_start
        tem_ckpt = model_dir / f"tem_{i_load}.pt"
        params_ckpt = model_dir / f"params_{i_load}.pt"

    paths = RunPaths(
        run=run_dir,
        train=run_dir / "train",
        model=model_dir,
        save=run_dir / "save",
        script=run_dir / "script",
        envs=utils.resolve_envs_path(run_dir),
    )

    # Validate checkpoint files exist
    _require_exists(tem_ckpt, f"Model checkpoint (iteration {i_load})")
    _require_exists(params_ckpt, f"Params checkpoint (iteration {i_load})")

    # Load params + apply overrides
    params = torch.load(params_ckpt, weights_only=False)
    params = utils.apply_overrides(params, settings.override)

    # Build model and load weights
    tem = model.Model(params)
    weights = torch.load(tem_ckpt, weights_only=False)
    tem.load_state_dict(weights)

    envs = [str(p) for p in paths.envs.glob("*")]
    i_start = i_load + 1
    return tem, envs, i_start, paths, params


def create_new(settings: RunSettings):
    """
    Create a new run: directories, params, model, and env list.

    Returns:
        (tem, envs, i_start, paths, params)
    """
    i_start = 0
    paths = _resolve_run_paths_for_new_run(settings)

    params = params_from_settings(settings)
    np.save(paths.save / "params", params)

    tem = model.Model(params)

    # Validate environment files exist
    for env_path in settings.envs:
        _require_exists(env_path, f"Environment file")

    envs = [str(p) for p in settings.envs]
    for env_file in set(envs):
        shutil.copy2(env_file, paths.envs / Path(env_file).name)

    return tem, envs, i_start, paths, params


def main() -> None:
    """Run TEM training."""
    settings = RunSettings()

    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)

    if settings.load_model:
        tem, envs, i_start, paths, params = load_existing(settings)
    else:
        tem, envs, i_start, paths, params = create_new(settings)

    def save_checkpoint(step: int) -> None:
        """Save model and params checkpoint."""
        torch.save(tem.state_dict(), paths.model / f"tem_{step}.pt")
        torch.save(tem.hyper, paths.model / f"params_{step}.pt")

    writer = SummaryWriter(utils.as_dir_str(paths.train))
    logger = utils.make_logger(utils.as_dir_str(paths.run))

    adam = torch.optim.Adam(tem.parameters(), lr=params["lr_max"])

    environments = [
        data.World(
            graph,
            randomise_observations=settings.randomise_observations,
            shiny=(params["shiny"] if np.random.rand() < params["shiny_rate"] else None),
        )
        for graph in np.random.choice(envs, params["batch_size"])
    ]
    visited = [[False for _ in range(env.n_locations)] for env in environments]
    walks = [
        env.generate_walks(
            params["n_rollout"] * np.random.randint(params["walk_it_min"], params["walk_it_max"]),
            1,
        )[0]
        for env in environments
    ]
    prev_iter = None

    for i in range(i_start, params["train_it"]):
        start_time = time.time()

        eta_new, lambda_new, p2g_scale_offset, lr, walk_length_center, loss_weights = model.parameter_iteration(i, params)
        tem.hyper["eta"] = eta_new
        tem.hyper["lambda"] = lambda_new
        tem.hyper["p2g_scale_offset"] = p2g_scale_offset
        for param_group in adam.param_groups:
            param_group["lr"] = lr

        chunk: list[list[list[Any]]] = []
        for env_i, walk in enumerate(walks):
            if len(walk) < params["n_rollout"]:
                environments[env_i] = data.World(
                    envs[np.random.randint(len(envs))],
                    randomise_observations=settings.randomise_observations,
                    shiny=(params["shiny"] if np.random.rand() < params["shiny_rate"] else None),
                )
                visited[env_i] = [False for _ in range(environments[env_i].n_locations)]
                walk = environments[env_i].generate_walks(
                    params["n_rollout"]
                    * np.random.randint(
                        walk_length_center - params["walk_it_window"] * 0.5,
                        walk_length_center + params["walk_it_window"] * 0.5,
                    ),
                    1,
                )[0]
                walks[env_i] = walk
                prev_iter[0].a[env_i] = None
                logger.info("Iteration %d: new walk length %d for batch %d", i, len(walk), env_i)

            for step in range(params["n_rollout"]):
                if len(chunk) < params["n_rollout"]:
                    chunk.append([[comp] for comp in walk.pop(0)])
                else:
                    for comp_i, comp in enumerate(walk.pop(0)):
                        chunk[step][comp_i].append(comp)

        for i_step, step in enumerate(chunk):
            chunk[i_step][1] = torch.stack(step[1], dim=0)

        forward = tem(chunk, prev_iter)

        loss = torch.tensor(0.0)
        plot_loss = 0
        for step in forward:
            step_loss = []
            for env_i, env_visited in enumerate(visited):
                if env_visited[step.g[env_i]["id"]]:
                    step_loss.append(loss_weights * torch.stack([l[env_i] for l in step.L]))
                else:
                    env_visited[step.g[env_i]["id"]] = True
            step_loss = torch.tensor(0) if not step_loss else torch.mean(torch.stack(step_loss, dim=0), dim=0)
            plot_loss = plot_loss + step_loss.detach().numpy()
            loss = loss + torch.sum(step_loss)

        adam.zero_grad()
        loss.backward(retain_graph=True)
        adam.step()
        prev_iter = [forward[-1].detach()]

        acc_p, acc_g, acc_gt = np.mean([[np.mean(a) for a in step.correct()] for step in forward], axis=0)
        acc_p, acc_g, acc_gt = [a * 100 for a in (acc_p, acc_g, acc_gt)]

        if settings.log_every > 0 and i % settings.log_every == 0:
            logger.info("Finished backprop iter %d in %.2f seconds.", i, time.time() - start_time)
            logger.info(
                "Loss: %.2f. <p_g> %.2f <p_x> %.2f <x_gen> %.2f <x_g> %.2f <x_p> %.2f <g> %.2f <reg_g> %.2f <reg_p> %.2f",
                loss.detach().numpy(),
                *plot_loss,
            )
            logger.info("Accuracy: <p> %.2f%% <g> %.2f%% <gt> %.2f%%", acc_p, acc_g, acc_gt)
            logger.info(
                "Parameters: <max_hebb> %.2f <eta> %.2f <lambda> %.2f <p2g_scale_offset> %.2f",
                np.max(np.abs(prev_iter[0].M[0].numpy())),
                tem.hyper["eta"],
                tem.hyper["lambda"],
                tem.hyper["p2g_scale_offset"],
            )
            logger.info("Weights: %s", [w for w in loss_weights.numpy()])
            logger.info(" ")

            # Log to TensorBoard
            scalars = [
                ("Losses/Total", loss.detach().numpy()),
                ("Losses/p_g", plot_loss[0]),
                ("Losses/p_x", plot_loss[1]),
                ("Losses/x_gen", plot_loss[2]),
                ("Losses/x_g", plot_loss[3]),
                ("Losses/x_p", plot_loss[4]),
                ("Losses/g", plot_loss[5]),
                ("Losses/reg_g", plot_loss[6]),
                ("Losses/reg_p", plot_loss[7]),
                ("Accuracies/p", acc_p),
                ("Accuracies/g", acc_g),
                ("Accuracies/gt", acc_gt),
            ]
            for tag, val in scalars:
                writer.add_scalar(tag, val, i)

        if settings.save_every > 0 and i % settings.save_every == 0:
            save_checkpoint(i)

    save_checkpoint(i)


if __name__ == "__main__":
    main()
