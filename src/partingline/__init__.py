"""partingline: parting-line analysis for mold design on exact OpenCascade B-rep."""

from .model import Model, load_model
from .analysis import Options, Result, analyze
from .pull import parse_pull, pull_from_face

__all__ = [
    "Model",
    "load_model",
    "Options",
    "Result",
    "analyze",
    "parse_pull",
    "pull_from_face",
]

__version__ = "0.1.0"
