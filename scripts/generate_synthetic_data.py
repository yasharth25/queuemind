"""Create approximately 60 days of clearly labelled QueueMind demo data."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_locations
from src.synthetic.generator import generate_synthetic_queue_data, validate_synthetic_queue_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SYNTHETIC QueueMind queue data.")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/queue_observations_synthetic.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = generate_synthetic_queue_data(load_locations(), "2026-01-01", args.days, seed=args.seed)
    validate_synthetic_queue_data(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(f"Wrote {len(data)} SYNTHETIC observations to {args.output}")


if __name__ == "__main__":
    main()
