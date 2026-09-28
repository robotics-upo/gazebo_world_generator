"""Compare maps produced by the Classic and Harmonic companion plugins."""

import math
import os
from pathlib import Path

import pytest
import yaml


def _read_map(base):
    metadata = yaml.safe_load(base.with_suffix(".yaml").read_text())
    header, dimensions, max_value, pixels = base.with_suffix(".pgm").read_bytes().split(b"\n", 3)
    assert header == b"P5" and max_value == b"255"
    width, height = map(int, dimensions.split())
    assert len(pixels) == width * height
    return metadata, width, height, pixels


def _pixel_at(metadata, width, height, pixels, x, y):
    resolution = metadata["resolution"]
    col = math.floor((x - metadata["origin"][0]) / resolution + 1e-9)
    row = height - 1 - math.floor((y - metadata["origin"][1]) / resolution + 1e-9)
    return pixels[row * width + col]


@pytest.mark.parametrize("suffix,expected_origin,expected_size", [
    ("MAP", [-5, -5, 0], (200, 200)),
    ("CONNECTED_MAP", [-11.5, -6, 0], (460, 240)),
    ("WAREHOUSE_MAP", [-9, -8, 0], (360, 320)),
])
def test_classic_harmonic_map_parity(suffix, expected_origin, expected_size):
    classic = os.environ.get(f"GWG_CLASSIC_{suffix}")
    harmonic = os.environ.get(f"GWG_HARMONIC_{suffix}")
    if not classic or not harmonic:
        pytest.skip(f"Set GWG_CLASSIC_{suffix} and GWG_HARMONIC_{suffix} to map base paths")
    classic_map = _read_map(Path(classic))
    harmonic_map = _read_map(Path(harmonic))
    classic_meta, width, height, pixels = classic_map
    harmonic_meta, h_width, h_height, h_pixels = harmonic_map
    for key in ("resolution", "origin", "mode", "occupied_thresh", "free_thresh"):
        assert classic_meta[key] == harmonic_meta[key]
    assert classic_meta["resolution"] == 0.05
    assert classic_meta["origin"] == expected_origin
    assert (width, height) == (h_width, h_height) == expected_size
    assert pixels == h_pixels
    assert pixels.count(254) > 0  # free
    assert pixels.count(0) > 0  # occupied wall boundary
    assert pixels.count(205) > 0  # unknown wall interior
    assert _pixel_at(classic_meta, width, height, pixels, 0, 0) == 254
    if suffix == "MAP":
        assert _pixel_at(classic_meta, width, height, pixels, 0.8, 0) == 0
    elif suffix == "WAREHOUSE_MAP":
        assert any(_pixel_at(classic_meta, width, height, pixels, x / 20, y / 20) == 0
                   for x in range(10, 31) for y in range(10, 31))
