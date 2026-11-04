import pandas as pd
import pytest
import sys

from src.evaluation.vision import TEMPLATE_CSV, evaluate_vision_labels, validate_vision_labels
from scripts.evaluate_vision import main


def test_vision_evaluation_uses_only_supplied_labels():
    labels = pd.DataFrame({"frame_id": ["f1", "f2", "f3"],
                           "ground_truth_queue_count": [2, 5, 1],
                           "predicted_queue_count": [2, 4, 3]})
    metrics, frame = evaluate_vision_labels(labels)
    assert metrics["labelled_frames"] == 3
    assert metrics["queue_count_mae"] == 1
    assert metrics["queue_count_rmse"] == pytest.approx((5 / 3) ** .5)
    assert metrics["exact_count_accuracy"] == pytest.approx(1 / 3)
    assert metrics["within_one_person_accuracy"] == pytest.approx(2 / 3)
    assert metrics["mean_signed_error"] == pytest.approx(1 / 3)
    assert metrics["max_absolute_error"] == 2
    assert frame.signed_error.tolist() == [0, -1, 2]


@pytest.mark.parametrize("bad", [
    pd.DataFrame(),
    pd.DataFrame({"frame_id": ["x", "x"], "ground_truth_queue_count": [1, 2], "predicted_queue_count": [1, 2]}),
    pd.DataFrame({"frame_id": ["x"], "ground_truth_queue_count": [-1], "predicted_queue_count": [1]}),
    pd.DataFrame({"frame_id": ["x"], "ground_truth_queue_count": [1.5], "predicted_queue_count": [1]}),
    pd.DataFrame({"frame_id": ["x"], "ground_truth_queue_count": [1], "predicted_queue_count": [float("nan")]}),
])
def test_invalid_label_csv_is_rejected(bad):
    with pytest.raises(ValueError):
        validate_vision_labels(bad)


def test_download_template_contains_only_schema_not_fake_results():
    assert TEMPLATE_CSV == "frame_id,ground_truth_queue_count,predicted_queue_count\n"


def test_standalone_vision_evaluation_command(monkeypatch, capsys):
    labels = pd.DataFrame({"frame_id": ["f1", "f2"],
                           "ground_truth_queue_count": [2, 4],
                           "predicted_queue_count": [3, 4]})
    monkeypatch.setattr(pd, "read_csv", lambda _: labels)
    monkeypatch.setattr(sys, "argv", ["evaluate_vision", "--labels", "labels.csv"])
    main()
    output = capsys.readouterr().out
    assert "Queue Count MAE: 0.5000 people" in output
    assert "Labelled frames: 2" in output
