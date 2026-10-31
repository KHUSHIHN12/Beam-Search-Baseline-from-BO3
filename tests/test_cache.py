"""test_cache.py - Unit tests for disk-backed EvaluationCache."""

import os
import tempfile
import pytest
from beampass.cache import EvaluationCache


@pytest.fixture
def temp_cache_file():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    if os.path.exists(path):
        os.unlink(path)
    yield path
    if os.path.exists(path):
        os.unlink(path)


def test_cache_miss_and_hit(temp_cache_file):
    """Verifies that missing keys return None and increment misses, hits return cached values."""
    cache = EvaluationCache(cache_file=temp_cache_file)
    assert cache.hits == 0
    assert cache.misses == 0

    # Miss
    val = cache.get("small_cnn", "default", ["canon", "CC"])
    assert val is None
    assert cache.misses == 1
    assert cache.hits == 0

    # Set
    cache.set(
        model="small_cnn",
        knob_config="default",
        sequence=["canon", "CC"],
        latency=21.5,
        final_mean=21.5,
        final_std=0.2,
    )

    # Hit
    val2 = cache.get("small_cnn", "default", ["canon", "CC"])
    assert val2 is not None
    assert val2["latency"] == 21.5
    assert cache.hits == 1
    assert cache.misses == 1


def test_cache_disk_persistence(temp_cache_file):
    """Cache data must persist across instances using the same file."""
    cache1 = EvaluationCache(cache_file=temp_cache_file)
    cache1.set("resnet18", "default", ["CC", "I"], latency=42.0)

    # New instance loading from same disk file
    cache2 = EvaluationCache(cache_file=temp_cache_file)
    hit = cache2.get("resnet18", "default", ["CC", "I"])
    assert hit is not None
    assert hit["latency"] == 42.0


def test_cache_clear(temp_cache_file):
    """Clear must wipe in-memory entries and remove the disk file."""
    cache = EvaluationCache(cache_file=temp_cache_file)
    cache.set("resnet18", "default", ["canon"], latency=60.0)
    assert len(cache) == 1
    assert os.path.exists(temp_cache_file)

    cache.clear()
    assert len(cache) == 0
    assert not os.path.exists(temp_cache_file)

