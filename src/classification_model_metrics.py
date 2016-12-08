def calc_ROC(targets, soft_predictions, plot_ROC=False):
    """
    determine ROC of SVM classifier

    :param targets: numpy vector of m targets
    :param soft_predictions: class probabilities
    :param plot_ROC: if true generate plot of ROC
    :return: array of [false pos. rate, false neg. rate]
    """

    import numpy as np
    from sklearn.metrics import roc_curve
    from matplotlib import pyplot as plt

    fpr, tpr, thresh = roc_curve(targets, soft_predictions)
    
    if (plot_ROC == True):
        plt.plot(fpr,tpr)
        plt.plot([0,1],[0,1],"r--",alpha=.5)
        plt.axis((-0.1, 1.1, -0.1, 1.1))
        plt.show()

    return [fpr,tpr]

def calc_AUC(targets, soft_predictions):
    """
    determine AUC of ROC of SVM classifier

    :param targets: numpy vector of m targets
    :param soft_predictions: class probabilities
    :return: AUC of ROC
    """

    import numpy as np
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(targets, soft_predictions)

    return auc
