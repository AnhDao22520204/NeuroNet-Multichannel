import numpy as np
from evaluation.sequence import aggregate_sequence_predictions


def test_overlapping_epoch_counted_once():
    # Same record, overlapping epoch 1 appears twice with compatible logits.
    logits = [np.array([[[3,0,0,0,0],[0,3,0,0,0]], [[0,4,0,0,0],[0,0,4,0,0]]], dtype=float)]
    labels = [np.array([[0,1],[1,2]], dtype=int)]
    records = [np.array([0,0], dtype=int)]
    indices = [np.array([[0,1],[1,2]], dtype=int)]
    y_true, y_pred = aggregate_sequence_predictions(logits, labels, records, indices)
    assert y_true.tolist() == [0,1,2]
    assert y_pred.tolist() == [0,1,2]
