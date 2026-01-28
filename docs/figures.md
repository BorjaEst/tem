# Figures

This document describes the torch_tem.figures package and enumerates the built-in registered figures (the figures you can address by stable name through the global registry).

## How to interpret figure outputs (episode vs aggregate)

Many built-in figures fall into one of two intent categories:

- Episode (contiguous): best viewed on a single contiguous rollout so temporal ordering and revisits can be inspected.
- Aggregate (coverage): best viewed on aggregated experience (multiple batches or episodes) to improve spatial occupancy and make rate maps or statistics meaningful.
- Static: depends primarily on the environment geometry or dataset composition and does not require long rollouts.

## Built-in figure inventory

The table below lists the figures that are registered by default.

| Figure                              | Summary |
| ----------------------------------- | ------- |
| [overview][mod_overview]            |         |
| [overview.rate_maps][mod_rate_maps] |         |

> TODO plots

| Figure                                              | Summary                                   |
| --------------------------------------------------- | ----------------------------------------- |
| [environment.layout][mod_env_layout]                | Static environment layout                 |
| [walk.trajectories][mod_walk_traj]                  | Trajectory overlay on map                 |
| [walk.statistics][mod_walk_stats]                   | Walk length/actions/visits/shiny stats    |
| [split.statistics][mod_split_stats]                 | Dataset composition histograms            |
| [coverage.occupancy_map][mod_occ_map]               | Per-location visit count and coverage     |
| [coverage.action_bias_map][mod_bias_map]            | Per-location action entropy (bias)        |
| [cells.place_rate_maps][mod_place_maps]             | Occupancy + grid of cell rate maps        |
| [cells.place_rate_maps.frequencies][mod_place_maps] | Same cells across all frequencies         |
| [cells.place_rate_maps.pathways][mod_place_maps]    | Same cells across pathways                |
| [cells.place_field_summary][mod_field_sum]          | Sparsity, peak, field-size histograms     |
| [cells.remapping_correlation][mod_remap]            | Env-by-env remapping correlation matrix   |
| [cells.spatial_autocorrelogram][mod_autocorr]       | Radial autocorrelogram vs distance        |
| [decoding.location_error_map][mod_decode_err]       | Ridge-decoding error per location         |
| [dynamics.path_integration_drift][mod_drift]        | L2 drift between g_inf and g_gen          |
| [dynamics.sequence_consistency][mod_seq]            | Cosine similarity of g_gen(t) vs g_inf    |
| [representation.freq_similarity][mod_freq]          | Frequency-by-frequency similarity heatmap |
| [memory.retrieval_error_by_location][mod_mem]       | Reconstruction MSE per location           |
| [uncertainty.calibration][mod_unc]                  | Uncertainty calibration curve             |

[mod_overview]: ../src/torch_tem/figures/modules/overview/observations.py
[mod_rate_maps]: ../src/torch_tem/figures/modules/overview/rate_maps.py
[mod_env_layout]: ../src/torch_tem/figures/modules/environment/layout.py
[mod_walk_traj]: ../src/torch_tem/figures/modules/walk/trajectories.py
[mod_walk_stats]: ../src/torch_tem/figures/modules/walk/statistics.py
[mod_split_stats]: ../src/torch_tem/figures/modules/split/statistics.py
[mod_occ_map]: ../src/torch_tem/figures/modules/coverage/occupancy_map.py
[mod_bias_map]: ../src/torch_tem/figures/modules/coverage/action_bias_map.py
[mod_place_maps]: ../src/torch_tem/figures/modules/cells/place_rate_maps.py
[mod_field_sum]: ../src/torch_tem/figures/modules/cells/place_field_summary.py
[mod_remap]: ../src/torch_tem/figures/modules/cells/remapping_correlation.py
[mod_autocorr]: ../src/torch_tem/figures/modules/cells/spatial_autocorrelogram.py
[mod_decode_err]: ../src/torch_tem/figures/modules/decoding/location_error_map.py
[mod_drift]: ../src/torch_tem/figures/modules/dynamics/path_integration_drift.py
[mod_seq]: ../src/torch_tem/figures/modules/dynamics/sequence_consistency.py
[mod_freq]: ../src/torch_tem/figures/modules/representation/freq_similarity.py
[mod_mem]: ../src/torch_tem/figures/modules/memory/retrieval_error_by_location.py
[mod_unc]: ../src/torch_tem/figures/modules/uncertainty/calibration.py
