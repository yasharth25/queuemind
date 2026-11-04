"""Calibratable normalized camera geometry shared by local vision sessions."""

from pathlib import Path

import yaml

from src.config import BASE_DIR


def load_vision_config(path: Path | None = None) -> dict:
    source = path or BASE_DIR / "config" / "vision.yaml"
    with source.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def validate_polygon(points) -> tuple[tuple[float, float], ...]:
    polygon = _points(points, minimum=3)
    area = sum(
        x * polygon[(index + 1) % len(polygon)][1]
        - polygon[(index + 1) % len(polygon)][0] * y
        for index, (x, y) in enumerate(polygon)
    )
    if abs(area) < 1e-6:
        raise ValueError("Queue ROI must enclose a nonzero area.")
    return polygon


def validate_line(points) -> tuple[tuple[float, float], tuple[float, float]]:
    line = _points(points, minimum=2)
    if len(line) != 2 or line[0] == line[1]:
        raise ValueError("A virtual line requires two distinct normalized points.")
    return line


def _points(points, minimum):
    try:
        values = tuple((float(x), float(y)) for x, y in points)
    except (TypeError, ValueError) as exc:
        raise ValueError("Coordinates must be normalized x/y pairs.") from exc
    if len(values) < minimum or any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in values):
        raise ValueError("Coordinates must be normalized x/y pairs in [0, 1].")
    return values
