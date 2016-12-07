def train_model(features, targets, model_filename):
    """
    train SVM classifier 

    :param features: numpy array of n features for m samples
    :param targets: numpy vector of m targets
    :param model_filename: filename to save the model to
    """

    import numpy as np
    from sklearn.svm import SVC
    import cPickle 

    # Define the model
    svm = SVC()
    
    # Train the model
    svm.fit(features, targets)

    # Save the model as an object
    with open(model_filename, 'wb') as f:
        cPickle.dump(svm,f)


def class_predict(test_features, model_filename):
    """
    binary classification based on SVM model

    :param test_features: numpy array of n features for m samples
    :param model_filename: filename containing SVM model
    :return: numpy vector of predicted class for m samples
    """

    import numpy as np
    from sklearn.svm import SVC
    import cPickle 

    # Load the model
    with open(model_filename, 'rb') as f:
        svm = cPickle.load(f)

    # Predict class of test samples
    class = svm.predict(test_features)

    return class
