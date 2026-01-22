"""Smoke tests for the figures subsystem.

Minimal tests to verify that core components can be instantiated and
produce expected outputs without crashing. Does not validate correctness,
only that the API contracts are satisfied.
"""

import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pytest

# Use non-interactive backend for headless environments
matplotlib.use('Agg')

from torch_tem.diagnostics.rollout_trace import TEMRolloutTrace
from torch_tem.figures import tem_overview
from torch_tem.figures.sinks import make_figure_path, save_pdf
from torch_tem.figures.style import DEFAULT_STYLE, StyleConfig


def create_dummy_trace(n_steps=50, batch_size=4, n_frequencies=3, n_place=16, n_grid=32, n_obs=45):
    """Create a minimal synthetic TEMRolloutTrace for testing."""
    return TEMRolloutTrace(
        n_steps=n_steps,
        batch_size=batch_size,
        o_predicted=np.random.rand(n_steps, batch_size, n_obs).astype(np.float32),
        o_true=np.random.rand(n_steps, batch_size, n_obs).astype(np.float32),
        p_inf=[np.random.rand(n_steps, batch_size, n_place).astype(np.float32) for _ in range(n_frequencies)],
        p_gen=[np.random.rand(n_steps, batch_size, n_place).astype(np.float32) for _ in range(n_frequencies)],
        g_inf=[np.random.rand(n_steps, batch_size, n_grid).astype(np.float32) for _ in range(n_frequencies)],
        g_gen=[np.random.rand(n_steps, batch_size, n_grid).astype(np.float32) for _ in range(n_frequencies)],
        location_ids=np.random.randint(0, 10, (n_steps, batch_size), dtype=np.int32),
        actions=np.random.randint(0, 4, (n_steps, batch_size), dtype=np.int32),
        p_xi=None,  # Optional field
        env_names=None,
    )


def test_style_config_instantiation():
    """Test that StyleConfig can be instantiated and applied."""
    style = StyleConfig(dpi=100, font_size=12)
    assert style.dpi == 100
    assert style.font_size == 12
    
    # Verify default style exists
    assert DEFAULT_STYLE.dpi == 150


def test_tem_overview_figure_creation():
    """Test that tem_overview.make_figure returns a Figure object."""
    trace = create_dummy_trace()
    fig = tem_overview.make_figure(trace, env_idx=0, freq_idx=0)
    
    # Verify it's a matplotlib Figure
    from matplotlib.figure import Figure
    assert isinstance(fig, Figure)
    
    # Verify it has the expected number of axes (2x2 grid)
    assert len(fig.axes) == 4


def test_save_pdf_creates_file():
    """Test that save_pdf creates a non-empty PDF file."""
    trace = create_dummy_trace(n_steps=10)
    fig = tem_overview.make_figure(trace)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "test_figure.pdf"
        save_pdf(fig, pdf_path)
        
        # Verify file was created and is non-empty
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0


def test_make_figure_path():
    """Test deterministic path generation."""
    base = Path("/logs/run_0")
    
    # Without step
    path = make_figure_path(base, "test_fig", extension="pdf")
    assert path == Path("/logs/run_0/figures/test_fig.pdf")
    
    # With step
    path = make_figure_path(base, "test_fig", step=1000, extension="pdf")
    assert path == Path("/logs/run_0/figures/test_fig_step1000.pdf")
    
    # With version
    path = make_figure_path(base, "test_fig", version="v1", extension="png")
    assert path == Path("/logs/run_0/figures/test_fig_version_v1.png")


def test_trace_select_env():
    """Test that select_env extracts a single environment correctly."""
    trace = create_dummy_trace(n_steps=10, batch_size=4)
    trace_single = trace.select_env(env_idx=1)
    
    assert trace_single.batch_size == 1
    assert trace_single.n_steps == trace.n_steps
    assert trace_single.o_predicted.shape == (10, 1, 45)


def test_trace_downsample_time():
    """Test that downsample_time reduces temporal dimension."""
    trace = create_dummy_trace(n_steps=100, batch_size=2)
    trace_ds = trace.downsample_time(stride=5)
    
    assert trace_ds.n_steps == 20  # 100 / 5
    assert trace_ds.batch_size == trace.batch_size
    assert trace_ds.o_predicted.shape[0] == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
