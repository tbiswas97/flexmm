"""Public API for FlexMM image segmentation and posterior analysis."""

from importlib import import_module

__all__ = ["Posterior", "SegmentationMap"]
__version__ = "0.1.0"


def __getattr__(name):
    """Load the two public classes without importing every backend eagerly."""
    modules = {
        "Posterior": "flexmm.posterior",
        "SegmentationMap": "flexmm.segmentation_map",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name]), name)
    globals()[name] = value
    return value
