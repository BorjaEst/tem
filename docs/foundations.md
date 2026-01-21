# Foundations: Types and Settings

This document contains background and longer-form notes that are intentionally kept out of in-code docstrings.

## `types.py`

The types module defines the core, shared vocabulary used throughout TEM.

Design constraints:

- Dependency-light: no imports from internal `torch_tem` modules.
- Types and dataclasses only.

Notes:

- A **multi-scale code** is represented as a `List[Tensor]`, one tensor per frequency module.
- Many modules use consistent shape notation:
  - `B`: batch size
  - `shape[f]`: per-frequency feature size
  - `S = sum(shape)`

## `settings.py`

The settings module defines Pydantic models for configuring TEM components.

Intended usage:

- `*Settings` classes are leaf-level configuration objects.
- Higher-level code composes these settings into full model/training configuration.

Documentation policy:

- In-code docstrings should say _what the setting controls_.
- Longer rationale (why default values are what they are, or how schedules interact) belongs in markdown under `docs/`.

## Handoff

- For docstring conventions and templates, see `docs/docstrings.md`.
- When adding substantial background (theory, derivations, long explanations), add it here (or another `docs/*.md`) and link from the relevant module docstring.
