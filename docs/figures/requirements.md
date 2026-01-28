# Figures Architecture Requirements

## Introduction

This document specifies the architecture pattern for a Python visualization library that separates:

- **plots**: low-level, axis-level plot primitives that draw onto a provided plotting target (e.g., a Matplotlib `Axes`).
- **figures**: high-level figure orchestration (figure creation, subplot/grid layout, styling/themes/templates, and composition of multiple plot primitives).

This separation follows the Figure–Axes separation popularized by Matplotlib and aligns with the Grammar of Graphics concept of layering and composition.

## 1. Purpose & Scope

- Define clear responsibilities, interfaces, and constraints for `plots/` and `figures/`.
- Establish requirements that improve modularity, testability, backend independence, and layout consistency.

### In Scope

- Function-level contracts for axis-level primitives.
- Figure-level orchestration responsibilities (layout, theming, composition, export policy).
- Return conventions and metadata needed for composition (legends, colorbars, shared scales).
- Testing strategy aligned with the separation.

### Out of Scope

- Implementing specific plot types or figure recipes.
- Choosing a specific rendering backend (Matplotlib, Plotly, Altair) as the only supported option.
- Full API reference documentation for end users.

## 2. Definitions

- **Axes**: A plotting surface within a figure (e.g., `matplotlib.axes.Axes`).
- **Figure**: A container for one or more Axes plus layout and figure-level elements (e.g., `matplotlib.figure.Figure`).
- **Plot primitive**: A low-level, axis-level function that draws onto a provided Axes/target.
- **Figure orchestration**: High-level logic that creates/manages figures, subplots, layout, styling, and plot composition.
- **Theme**: A named set of styling defaults (fonts, sizes, colors, grid rules).
- **Template**: A reusable figure-level configuration that can include theme + layout + export settings.
- **Guide**: A visual explanation of encodings, typically legends and colorbars.
- **Resolved style**: Concrete rendering parameters passed to primitives (e.g., `color="#4C78A8"`, `lw=2`).

## 3. Requirements, Constraints & Guidelines

### Core Architecture Requirements

- **REQ-001**: WHEN a plot primitive is called, THE SYSTEM SHALL draw only on the provided target (e.g., Axes) and SHALL NOT create a Figure.
- **REQ-002**: WHEN a plot primitive is called, THE SYSTEM SHALL NOT create or modify subplot grids, constrained layouts, or global layout settings.
- **REQ-003**: WHEN a plot primitive is called, THE SYSTEM SHALL NOT mutate global styling state (e.g., Matplotlib `rcParams`) as a side effect.
- **REQ-004**: WHEN a figure-level function is called, THE SYSTEM SHALL be able to create and manage Figure objects and subplot/grid layouts.
- **REQ-005**: WHEN a figure-level function composes multiple primitives, THE SYSTEM SHALL manage consistent guides (legends/colorbars) across axes.
- **REQ-006**: WHEN a figure-level function applies styling, THE SYSTEM SHALL apply theme/template policies in `figures/` and pass resolved style parameters into `plots/`.

### Composition and Return-Value Requirements

- **REQ-010**: WHEN a plot primitive produces artists/traces, THE SYSTEM SHALL return handles sufficient for later composition (legend entries, colorbar mappables, etc.).
- **REQ-011**: WHEN a plot primitive produces multiple handles, THE SYSTEM SHALL return a structured result object (e.g., a dataclass or NamedTuple) rather than an ambiguous tuple.
- **REQ-012**: WHEN a plot primitive uses an encoded color scale (continuous or categorical), THE SYSTEM SHALL expose the information needed for figure-level guide creation (e.g., a mappable, normalization, and/or labels).

### Dependency and Import Constraints

- **CON-001**: The `plots/` package SHALL NOT import from `figures/`.
- **CON-002**: The `figures/` package MAY import from `plots/`.
- **CON-003**: Shared helpers (data validation, color utilities) SHALL live in `utils/` (or another shared module) to avoid circular dependencies.

### Backend Independence Requirements

- **REQ-020**: WHEN a plot primitive is executed in a headless environment, THE SYSTEM SHALL be runnable without requiring interactive UI state.
- **REQ-021**: WHEN a figure-level function is executed, THE SYSTEM SHALL centralize export policy (DPI, size, background, file writing) in `figures/`.
- **GUD-001**: Plot primitives SHOULD accept an explicit target object rather than using global state (e.g., no implicit `plt.gca()` usage).

### Layout Consistency Requirements

- **REQ-030**: WHEN multiple axes are created by a figure-level function, THE SYSTEM SHALL apply a single, consistent layout engine and spacing policy.
- **REQ-031**: WHEN a composite figure includes multiple subplots, THE SYSTEM SHALL define a deterministic policy for shared vs independent scales.
- **REQ-032**: WHEN a composite figure includes a legend and/or colorbar, THE SYSTEM SHALL define a deterministic policy for placement and sizing.

### API Usability Guidelines

- **GUD-010**: Plot primitives SHOULD be small, single-purpose, and composable.
- **GUD-011**: Figure-level APIs SHOULD provide “recipes” for common multi-panel compositions while delegating rendering to `plots/`.
- **GUD-012**: Both layers SHOULD validate inputs and fail fast with clear exceptions; heavy data transformation SHOULD be placed in `utils/data.py` or figure-level preprocessing.

## 4. Interfaces & Data Contracts

### Plot Primitive Interface (Conceptual)

All plot primitives SHALL follow this conceptual interface:

```python
def primitive(ax, data, *, mapping=None, style=None, label=None, **kwargs):
  """Draw on `ax` and return handles and metadata for composition."""
  ...
```

### Structured Result Objects

Plot primitives that create multiple artists SHOULD return a structured result:

```python
from dataclasses import dataclass
from typing import Any, Sequence, Optional

@dataclass(frozen=True)
class HeatmapResult:
    image: Any
    mappable: Any
    vmin: Optional[float]
    vmax: Optional[float]

@dataclass(frozen=True)
class HistResult:
    patches: Sequence[Any]
    bin_edges: Sequence[float]
```

### Figure-Level Interface (Conceptual)

Figure-level functions manage layout, theming, composition, and export:

```python
def make_figure(*, template=None, theme=None, size=None):
  """Create and return a Figure and Axes grid plus orchestration metadata."""
  ...

def compose(*, template=None, theme=None):
  """Assemble primitives into a multi-axes figure with shared guides."""
  ...
```

## 5. Acceptance Criteria

- **AC-001**: Given an existing Axes, When `plots.line(ax, ...)` is called, Then no new Figure is created and the line is added to the provided Axes.
- **AC-002**: Given a clean global style state, When a plot primitive is called, Then global style (e.g., `rcParams`) remains unchanged after the call.
- **AC-003**: Given a multi-panel figure created by a figure-level function, When multiple primitives are composed, Then the layout spacing policy is applied consistently across all axes.
- **AC-004**: Given two subplots with the same categorical mapping, When composed via a figure-level function, Then a single combined legend is produced according to the figure-level legend policy.
- **AC-005**: Given two heatmaps intended to share a color scale, When composed via a figure-level function with shared scale enabled, Then both axes use the same normalization and a single colorbar is rendered.

## 6. Test Automation Strategy

- **Test Levels**
  - Unit: `plots/*` primitives
  - Integration: `figures/*` layout/theme/composition
  - Optional E2E: smoke tests that render and compare basic image properties
- **Frameworks**
  - Preferred: `pytest`
  - Optional: image regression tools (only if already used in the project)
- **Test Data Management**
  - Use small deterministic arrays/dataframes.
  - Avoid randomness unless seeded.
- **CI/CD Integration**
  - Run headless rendering tests in CI.
  - Fail fast on import cycles and style-state mutation.

## 7. Rationale & Context

- Separating primitives from orchestration reduces coupling and enables reuse of the same primitive across many layouts.
- Testing becomes simpler: primitives can be unit-tested on a single Axes; orchestration can be integration-tested for layout/guide policies.
- Backend independence improves because primitives do not rely on global, interactive state and figure-level export settings are centralized.
- Layout consistency improves because there is a single authority for spacing, shared scales, and guide placement.

## 8. Dependencies & External Integrations

### External Systems

- **EXT-001**: Rendering backend(s) (e.g., Matplotlib) - provides Figure/Axes primitives and export.

### Technology Platform Dependencies

- **PLT-001**: Python runtime - required for library execution and testing.

## 9. Examples & Edge Cases

```python
# Axis-level primitive usage
fig, ax = plt.subplots()
artist = plots.line(ax, x=[0, 1], y=[0, 1], label="identity")

# Figure-level composition usage
fig = figures.compose(

    panels=[
        lambda ax: plots.scatter(ax, x=a, y=b, label="A"),
        lambda ax: plots.scatter(ax, x=c, y=d, label="B"),
    ],
    layout=(1, 2),
    legend="shared",
    theme="paper",
)
```

Edge cases to handle:

- Primitives called with an Axes that already contains artists.
- Composition of primitives that set conflicting axis limits.
- Shared vs independent normalization for heatmaps.
- Legend merging when labels collide.

## 10. Validation Criteria

- No plot primitive creates figures or subplots.
- No plot primitive mutates global style state.
- Figure-level functions produce deterministic layout and guide policies.
- Import graph respects `plots` → `figures` direction constraint.

## 11. Related Specifications / Further Reading

- Matplotlib Figure–Axes separation (conceptual reference)
- Seaborn axes-level vs figure-level API pattern (conceptual reference)
- Plotly traces vs `Figure` templates/subplots (conceptual reference)
- Altair layering and composition (Grammar of Graphics) (conceptual reference)
