from __future__ import annotations

import functools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from make_fixtures import GENERATED, build_all  # noqa: E402

from partingline.analysis import Options, analyze  # noqa: E402
from partingline.model import load_model  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def generated_fixtures():
    """(Re)build the cadquery fixtures once per test session."""
    return build_all(GENERATED)


@functools.lru_cache(maxsize=None)
def model_for(name: str):
    return load_model(GENERATED / f"{name}.step")


@functools.lru_cache(maxsize=None)
def run(name: str, pull=(0.0, 0.0, 1.0), vertical: str = "top"):
    model = model_for(name)
    return model, analyze(model, pull, Options(vertical=vertical))
