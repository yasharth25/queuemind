"""Evaluate a manually labelled queue-count CSV without Streamlit or a camera."""

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.vision import evaluate_vision_labels


def main():
    parser = argparse.ArgumentParser(description="Evaluate real labelled queue-count frames.")
    parser.add_argument("--labels", type=Path, required=True, help="CSV with frame_id and ground-truth/predicted queue counts")
    parser.add_argument("--output-csv", type=Path, help="Optional per-frame error CSV path")
    args = parser.parse_args()
    try:
        metrics, frame = evaluate_vision_labels(pd.read_csv(args.labels))
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Labelled frames: {metrics['labelled_frames']}")
    print(f"Queue Count MAE: {metrics['queue_count_mae']:.4f} people")
    print(f"Queue Count RMSE: {metrics['queue_count_rmse']:.4f} people")
    print(f"Exact Count Accuracy: {metrics['exact_count_accuracy']:.2%}")
    print(f"Within ±1 Person Accuracy: {metrics['within_one_person_accuracy']:.2%}")
    print(f"Mean Signed Error: {metrics['mean_signed_error']:.4f} people")
    print(f"Max Absolute Error: {metrics['max_absolute_error']} people")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output_csv, index=False)
        print(f"Per-frame errors written to {args.output_csv}")


if __name__ == "__main__":
    main()
