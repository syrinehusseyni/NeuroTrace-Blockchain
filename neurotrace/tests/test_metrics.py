import numpy as np

from evaluation.metrics import classification_report, confusion_matrix, threshold_table

NAMES = ["a", "b", "c"]
Y_TRUE = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
Y_PRED = [0, 0, 1, 1, 1, 2, 2, 2, 2, 0]


def test_confusion_matrix():
    cm = confusion_matrix(Y_TRUE, Y_PRED, 3)
    assert cm.tolist() == [[2, 1, 0], [0, 2, 1], [1, 0, 3]]


def test_report_values():
    rep = classification_report(confusion_matrix(Y_TRUE, Y_PRED, 3), NAMES)
    assert rep["n_images"] == 10 and abs(rep["accuracy"] - 0.7) < 1e-9
    a = rep["per_class"]["a"]
    assert abs(a["precision"] - 2 / 3) < 1e-9 and abs(a["recall"] - 2 / 3) < 1e-9
    assert rep["per_class"]["c"]["support"] == 4
    assert abs(rep["macro_avg"]["recall"] - (2 / 3 + 2 / 3 + 3 / 4) / 3) < 1e-9


def test_empty_class_does_not_divide_by_zero():
    cm = np.array([[3, 0], [0, 0]])
    rep = classification_report(cm, ["x", "y"])
    assert rep["per_class"]["y"]["f1"] == 0.0


def test_threshold_table():
    correct = [True, True, False, True]
    conf = [0.99, 0.95, 0.60, 0.55]
    margin = [0.9, 0.2, 0.1, 0.5]
    rows = threshold_table(correct, conf, margin, [0.9, 0.5], margin_min=0.3)
    by = {(r["confidence_min"], r["margin_min"]): r for r in rows}
    assert by[(0.9, None)]["accepted"] == 2 and by[(0.9, None)]["accuracy_of_accepted"] == 1.0
    assert by[(0.9, 0.3)]["accepted"] == 1
    assert by[(0.5, None)]["errors_accepted"] == 1 and by[(0.5, None)]["coverage"] == 1.0
