import os
import sys
sys.path.insert(0, os.path.abspath('./src/'))


def test_model_train_predict():
    from classification_model import train_model
    from classification_model import class_predict
    import numpy as np
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    iris = datasets.load_iris()
    x_train, x_test, y_train, y_test = \
        train_test_split(iris.data, iris.target, test_size=0.2)

    train_model(x_train, y_train, 'test_model.pkl')

    assert os.path.isfile('test_model.pkl')

    y_pred = class_predict(x_test, 'test_model.pkl')
    misclassification = len(np.nonzero(y_pred - y_test)) / len(y_test)
    assert(misclassification < 0.1)

    os.system("rm test_model.pkl")
