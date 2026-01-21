# Docstrings and Comments (Google Style)

This project uses Google-style docstrings with a strong bias toward _right-sized_ documentation.

## Principles

1. **Put information where it belongs**
   - Module docstring: purpose + cross-cutting conventions.
   - Class docstring: responsibility + invariants/state semantics.
   - Method/function docstring: API contract.
   - Inline comments: only non-obvious _why_.

2. **Right-size docstrings**
   - If a docstring reads like a design doc, move most of it to markdown under `docs/`.
   - Prefer short summaries and precise contracts.

3. **Avoid duplication**
   - Don’t restate type hints.
   - Don’t repeat module-level conventions in every method.

## Templates

### Module docstring

Keep to ~5–20 lines.

```python
"""One-line summary.

Short paragraph describing purpose and where it sits in the system.

Conventions:
    - List the few conventions that materially affect usage.
"""
```

### Class docstring

Keep to ~5–15 lines.

```python
class Foo:
    """What this class does.

    Notes:
        - Invariants and gotchas.
    """
```

Use `Attributes:` only when it adds semantic meaning that is not obvious from type hints.

### Function / method docstring

```python
def bar(x: Tensor) -> Tensor:
    """Compute bar from x.

    Args:
        x: Meaningful description. Include shape *only when needed*.

    Returns:
        Meaningful description.

    Raises:
        ValueError: Only when the function can raise it.
    """
```

### Properties

One line is typically enough.

```python
@property
def foo(self) -> int:
    """Return foo."""
```

## Tensor Shape Notation

Use consistent notation across TEM modules:

- `B`: batch size
- `n_freq`: number of frequency modules
- `shape[f]`: feature size of frequency module `f`
- `S = sum(shape)`: flattened size of a multi-scale code

When a shape is non-obvious or commonly misused, include it in the docstring.

Examples:

- Multi-scale code: `List[Tensor]` with tensors shaped `(B, shape[f])`.
- Memory matrix: `(B, S, S)`.

## Inline Comments

Use comments only for:

- Math identities or transformations that are not obvious.
- Masking/gating logic.
- “Legacy parity” decisions.

Avoid:

- Comments that narrate the code.
- TODOs that belong in issues (prefer moving intent into `Notes:`).
