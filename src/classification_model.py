def train_model(features, targets, model_filename):
    """
    train SVM classifier 

    :param features: numpy array of n features for m samples
    :param targets: numpy vector of m targets
    :param model_filename: filename to save the model to
    """

    import numpy as np
    from sklearn.svm import SVC
    import pickle  
    from sklearn.model_selection import GridSearchCV

    # Define the hyperparameter options
    params = [{'kernel': ['rbf'], 'gamma': [1e-5, 1e-4, 1e-3, 1e-2],
               'C': [0.1, 1, 10, 100, 1000]},
              {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]}]

    # Define the model
    svm = SVC()

    # Grid search to optimize hyperparameters
    clf = GridSearchCV(svm, params)
    
    # Train the model
    clf.fit(features, targets)

    # Save the model as an object
    with open(model_filename, 'wb') as f:
        pickle.dump(clf.best_estimator_,f)


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
    with open(model_filename, 'rb') as f:
        clf = pickle.load(f)

    # Predict class of test samples
    class_pred = clf.predict(test_features)

    return class_pred
