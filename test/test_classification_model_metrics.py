import os
import sys
sys.path.insert(0, os.path.abspath('./src/'))


def test_calc_accuracy():
    from classification_model_metrics import calc_accuracy
    true = [1, 1, -1, 1, 1]
    pred = [1, -1, -1, 1, 1]

    actual = calc_accuracy(true, pred)
    expected = 4/5

    assert actual == expected


def test_roc_auc():
    from classification_model_metrics import calc_ROC
    from classification_model_metrics import calc_AUC
    import numpy as np

    targets = [0, 1, 0, 1]
    soft_predictions = [0.1, 0.9, 0.2, 0.8]
    #results = calc_ROC(targets, soft_predictions, False)
    auc = calc_AUC(targets, soft_predictions)
    assert auc == 1


def test_gen_confusion_matrix():
    from classification_model_metrics import gen_confusion_matrix
    import numpy as np

    targets = [0, 1, 0, 1]
    predictions = [0, 1, 0, 0]

    actual = gen_confusion_matrix(targets, predictions, ['1', '2'], False)
    expected = [[2, 0], [1, 1]]

    assert np.array_equal(actual, expected)
