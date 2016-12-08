import os
import sys
sys.path.insert(0, os.path.abspath('./src/'))

def test_roc_auc():
    from classification_model_metrics import calc_ROC
    from classification_model_metrics import calc_AUC
    import numpy as np

    targets = [0,1,0,1]
    soft_predictions = [0.1,0.9,0.2,0.8]
    #results = calc_ROC(targets,soft_predictions,False)
    auc = calc_AUC(targets,soft_predictions)
    assert(auc == 1)
