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

    accuracy = 100*np.sum(np.subtract(targets, predictions) == 0) / len(
        targets)

    return accuracy


def calc_f1_score(targets, predictions):
    """
    determine F1 score

    :param targets: numpy vector of m true targets
    :param predictions: numpy vector of m predicted targets
    :return: F1 score
    """
    import numpy as np

    true_pos = np.sum((targets == 1) & (predictions == 1)) / np.sum(targets == 1)
    false_pos = np.sum((targets == -1) & (predictions == 1)) / np.sum(targets == -1)
    false_neg = np.sum((targets == 1) & (predictions == -1)) / np.sum(targets == 1)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
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
    import numpy as np
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(targets, predictions, labels=None,
                          sample_weight=None)

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

    create_dir(outfile)
    plt.savefig(outfile)
    plt.clf()


def plot_features(features, targets, labels):
    """
    visualize 2D features for samples from 2 known classes

    :param features: N x 2 sets of feature values
    :param targets: target labels corresponding to each set of features
    :param labels: labels for each feature
    """
    import numpy as np
    import matplotlib.pyplot as plt

    target_types = np.unique(targets)
    if len(target_types) > 2:
        msg = 'ERROR [plot_features] Function only compatible with 2 targets.'
        logging.error(msg)
        print(msg)
        sys.exit()

    if np.shape(features)[1] > 2:
        msg = 'ERROR [plot_features] Function only compatible with 2 features.'
        logging.error(msg)
        print(msg)
        sys.exit()

    if np.shape(features)[0] != len(targets):
        msg = 'ERROR [plot_features] Mismatch between number of target ' \
              'labels and feature sets.'
        logging.error(msg)
        print(msg)
        sys.exit()

    features0 = features[targets == target_types[0], :]
    features1 = features[targets == target_types[1], :]

    h0 = plt.scatter(features0[:, 0], features0[:, 1], marker='o', c='red',
                     label=target_types[0])
    h1 = plt.scatter(features1[:, 0], features1[:, 1], marker='o', c='blue',
                     label=target_types[1])
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(handles=[h0, h1])
    plt.show()
