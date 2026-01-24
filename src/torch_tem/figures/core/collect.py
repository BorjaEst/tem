"""Generic trace collection from event streams.

Provides the collect_trace utility that consumes an event iterator
and a TraceExtractor to produce plot-ready traces.
"""

from __future__ import annotations

from typing import Iterator, Optional

from torch_tem.figures.core.types import EventT, TraceExtractor, TraceT


def collect_trace(
    stream: Iterator[EventT],
    extractor: TraceExtractor[EventT, TraceT],
    *,
    max_steps: Optional[int] = None,
    downsample_stride: int = 1,
) -> TraceT:
    """Collect a plot-ready trace from an event stream.

    Iterates through the event stream, passes sampled events to the extractor,
    and returns the finalized trace.

    Args:
        stream: Iterator yielding events (e.g., RolloutEvent objects).
        extractor: TraceExtractor that builds the trace from events.
        max_steps: Maximum number of events to observe (None = observe all).
        downsample_stride: Keep every stride-th event (1 = keep all).

    Returns:
        Finalized trace from the extractor.

    Raises:
        ValueError: If no events were collected (stream was empty or max_steps=0).

    Example:
        >>> from torch_tem.model import RolloutStream
        >>> from torch_tem.diagnostics.extractors import ModelTraceExtractor
        >>> rollout = RolloutStream(model, walk)
        >>> extractor = ModelTraceExtractor()
        >>> trace = collect_trace(rollout.iter_events(), extractor, max_steps=100)
    """
    step_count = 0
    collected_count = 0

    for event in stream:
        # Sample event based on stride
        if step_count % downsample_stride == 0:
            extractor.observe(event)
            collected_count += 1

        step_count += 1

        # Stop if max_steps reached (counts observed events, not collected)
        if max_steps is not None and step_count >= max_steps:
            break

    if collected_count == 0:
        raise ValueError("collect_trace: no events were collected (empty stream or max_steps=0)")

    return extractor.finalize()
