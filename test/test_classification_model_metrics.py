import os
import sys
sys.path.insert(0, os.path.abspath('./src/'))


def test_calc_accuracy():
    from classification_model_metrics import calc_accuracy
    true = [1, 1, -1, 1, 1]
    pred = [1, -1, -1, 1, 1]

    actual = calc_accuracy(true, pred)
    expected = 100*4/5

    assert actual == expected


def test_roc_auc():
    from classification_model_metrics import calc_ROC
    from classification_model_metrics import calc_AUC
    import numpy as np

    targets = [-1, 1, -1, 1]
    soft_predictions = [0.1, 0.9, 0.2, 0.8]
    auc = calc_AUC(targets, soft_predictions)
    assert auc == 1

def test_f1_score():
    from classification_model_metrics import calc_f1_score
    import numpy as np

    targets = np.array([1, 1, 1, 1, -1, -1, -1, -1])
    predictions = np.array([1, 1, -1, 1, 1, -1, -1, -1])
    f1 = calc_f1_score(targets, predictions)
    assert f1 == 0.75


def test_gen_confusion_matrix():
    from classification_model_metrics import gen_confusion_matrix
    import numpy as np

    targets = [-1, 1, -1, 1]
    predictions = [-1, 1, -1, -1]

    actual = gen_confusion_matrix(targets, predictions, ['1', '2'], False)
    expected = [[2, 0], [1, 1]]

    assert np.array_equal(actual, expected)
