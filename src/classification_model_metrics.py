import sys
import logging


def calc_accuracy(targets, predictions):
    """
    determine classification accuracy

    :param targets: numpy vector of m true targets
    :param predictions: numpy vector of m predicted targets
    :return: classification accuracy
    """
    import numpy as np

    try:
        accuracy = 100*np.sum(np.subtract(targets, predictions) == 0) / len(
            targets)
    except ValueError as err:
        msg = 'ERROR [calc_accuracy] mismatch between target array size and ' \
              'predictions array size: {0}'.format(err)
        print(msg)
        logging.error(msg)
        sys.exit()

    return accuracy


def calc_f1_score(targets, predictions):
    """
    determine F1 score

    :param targets: numpy vector of m true targets
    :param predictions: numpy vector of m predicted targets
    :return: F1 score
    """
    import numpy as np
    #import warnings
    #warnings.filterwarnings("error")

    true_pos = np.sum((targets == 1) & (predictions == 1)) / np.sum(targets == 1)
    false_pos = np.sum((targets == -1) & (predictions == 1)) / np.sum(targets == -1)
    false_neg = np.sum((targets == 1) & (predictions == -1)) / np.sum(targets == 1)
    try:
        precision = true_pos / (true_pos + false_pos)
    except RuntimeWarning:
        msg = 'Summation of TP + FP = 0, precision metric not reliable'
        logging.debug(msg)
        precision = 0

    try:
        recall = true_pos / (true_pos + false_neg)
    except RuntimeWarning:
        msg = 'Summation of TP + FN = 0, recall metric not reliable'
        logging.debug(msg)
        recall = 0

    if (precision == 0) | (recall == 0):
        print('F1 Score is not reliable as the number of TP, FP, or FN is 0')
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return f1


def calc_ROC(targets, soft_predictions, plot_ROC=False, outfile='./roc.png'):
    """
    determine ROC of SVM classifier

    :param targets: numpy vector of m targets
    :param soft_predictions: class probabilities
    :param plot_ROC: if true generate plot of ROC
    :param outfile: file to output ROC plot if verbose
    :return: array of [false pos. rate, false neg. rate]
    """
    import numpy as np
    from sklearn.metrics import roc_curve

    fpr, tpr, thresh = roc_curve(targets, soft_predictions)
    
    if plot_ROC:
        msg = '[calc_ROC] Saving ROC curve to: %s' % outfile
        logging.info(msg)
        print(msg)

        from matplotlib import pyplot as plt
        from accessory import create_dir
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], "r--", alpha=.5)
        plt.axis((-0.01, 1.01, -0.01, 1.01))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title('ROC for SVM Model on Test Set')

        create_dir(outfile)
        plt.savefig(outfile)
        plt.clf()

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


def gen_confusion_matrix(targets, predictions, classes, verb=False,
                         outfile='./confusion.png'):
    """
    generate confusion matrix to evaluate classification accuracy

    :param targets: numpy vector of m true targets
    :param predictions: numpy vector of m predicted targets
    :param classes: classification target names
    :param verb: verbose mode to visualize confusion matrix, default False
    :param outfile: file to output confusion matrix figure if verbose
    :return: confusion matrix
    """
    from sklearn.metrics import confusion_matrix

    try:
        cm = confusion_matrix(targets, predictions, labels=None,
                              sample_weight=None)
    except ValueError as err:
        msg = 'ERROR [gen_confusion_matrix] Mismatch between target array ' \
              'size and predictions array size: {0}'.format(err)
        print(msg)
        logging.error(msg)
        sys.exit()

    if verb:
        msg = '[gen_confusion_matrix] Saving confusion matrix to: %s' % outfile
        logging.info(msg)
        print(msg)
        plot_confusion_matrix(cm, classes, outfile=outfile)

    return cm


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          outfile='./cm.png'):
    """
    plots the confusion matrix (directly adapted from sklearn example)

    :param cm: confusion matrix
    :param classes: classification target names
    :param normalize: enable to normalize confusion matrix values
    :param title: plot title, default='Confusion Matrix'
    :param outfile: file to output confusion matrix figure if verbose
    """
    import numpy as np
    import itertools
    import matplotlib.pyplot as plt
    from accessory import create_dir

    if len(classes) != cm.shape[0]:
        msg = 'ERROR [plot_confusion_matrix] Mismatch between number of ' \
              'specified classes and number of total targets. ' \
              '(%d classes and %d targets)' % (len(classes), cm.shape[0])
        print(msg)
        logging.error(msg)
        sys.exit()

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

    plt.axis('tight')
    plt.ylabel('True')
    plt.xlabel('Predicted')

    create_dir(outfile)
    plt.savefig(outfile)
    plt.clf()
