import os
import sys
import logging


def train_model(features, targets, model_filename):
    """
    train SVM classifier

    :param features: numpy array of n features for m samples
    :param targets: numpy vector of m targets
    :param model_filename: filename to save the model to
    :return: svm model with best cross-validation results
    """
    from sklearn.svm import SVC
    import pickle
    from sklearn.model_selection import GridSearchCV

    msg = 'Training the SVM w/ CV gridsearch to find best hyperparams'
    logging.info(msg)

    # Define the hyperparameter options
    params = [{'kernel': ['rbf'], 'gamma': [1e-5, 1e-4, 1e-3, 1e-2],
               'C': [0.1, 1, 10, 100, 1000]},
              {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]}]

    # Define the model
    svm = SVC(probability=True)

    # Grid search to optimize hyperparameters
    clf = GridSearchCV(svm, params)

    # Train the model
    clf.fit(features, targets)

    # Save the model as an object
    try:
        with open(model_filename, 'wb') as f:
            pickle.dump(clf.best_estimator_, f)
    except FileNotFoundError:
        msg = 'ERROR [train_model] issue saving model object to %s' \
            % model_filename
        print(msg)
        logging.error(msg)
        sys.exit()

    msg = 'SVM w/ best CV performance saved as: %s' % model_filename
    logging.info(msg)

    return clf.best_estimator_


def class_predict(test_features, model_filename):
    """
    binary classification based on SVM model

    :param test_features: numpy array of n features for m samples
    :param model_filename: filename containing SVM model
    :return: numpy vector of predicted class for m samples
    """
    import numpy as np
    from sklearn.svm import SVC
    import pickle

    # Load the model
    try:
        with open(model_filename, 'rb') as f:
            clf = pickle.load(f)
    except FileNotFoundError:
        msg = 'ERROR [class_predict] %s is not a valid model_filename. \n' \
              'Ensure that you have previously trained the model by ' \
              'runnning main.py with train=True \n ' \
              'Exiting script...' % model_filename
        print(msg)
        logging.error(msg)
        sys.exit()

    # Predict class of test samples
    class_pred = clf.predict(test_features)

    return class_pred


def split_data(feature_array, target_array, split=False):
    """
    split data into training and testing (test=train data if split is false)

    :param feature_array: N x M array of M features per N datasets
    :param target_array: N targets associated with each feature set
    :param split: enable to split data into separate train and test
    :return: training and test feature sets with corresponding targets
    """
    from sklearn.model_selection import train_test_split
    import numpy as np

    msg = 'Split data into training and test: %s\n' % split
    logging.info(msg)
    print(msg)

    if split:
        # Split data in to training and testing (best practice)
        class_diff = False
        # Ensure training or test data don't have uniform class
        while not class_diff:
            x_train, x_test, y_train, y_test \
                = train_test_split(feature_array, target_array, test_size=0.3)
            if (np.std(y_train) != 0) & (np.std(y_test) != 0):
                class_diff = True
    else:
        # Use same data to train and test SVM
        x_train = feature_array
        y_train = target_array
        x_test = feature_array
        y_test = target_array

    return x_train, x_test, y_train, y_test
