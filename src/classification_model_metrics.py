import sys
import logging


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
    
    if plot_ROC == True:
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], "r--", alpha=.5)
        plt.axis((-0.01, 1.01, -0.01, 1.01))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title('ROC for SVM Model on Test Set')
        plt.show()

    return [fpr, tpr]


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


def gen_confusion_matrix(targets, predictions, classes, verb=False):
    """
    generate confusion matrix to evaluate classification accuracy

    :param targets: numpy vector of m true targets
    :param predictions: numpy vector of m predicted targets
    :param classes: classification target names
    :param verb: verbose mode to visualize confusion matrix, default False
    :return: confusion matrix
    """
    import numpy as np
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(targets, predictions, labels=None,
                          sample_weight=None)

    msg = '[gen_confusion_matrix] Calculating confusion matrix.'
    logging.debug(msg)
    if verb:
        print(msg)
        plot_confusion_matrix(cm, classes)

    return cm


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix'):
    """
    plots the confusion matrix (directly adapted from sklearn example)

    :param cm: confusion matrix
    :param classes: classification target names
    :param normalize: enable to normalize confusion matrix values
    :param title: plot title, default='Confusion Matrix'
    """
    import numpy as np
    import itertools
    import matplotlib.pyplot as plt

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()
