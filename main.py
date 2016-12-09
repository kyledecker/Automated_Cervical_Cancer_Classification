import os
import sys
import logging
import numpy as np
sys.path.insert(0, os.path.abspath('./src/'))


if __name__ == "__main__":
    from preprocess import read_tiff, rgb_histogram, rgb_preprocess
    from feature_extraction import *
    from classification_model import *
    from sklearn.model_selection import train_test_split
    from classification_model_metrics import *

    verb = False
    train = True

    if (train == True):
        train_files = os.listdir('./TrainingData')
        path = os.getcwd() + '/TrainingData/'
        num_features = 2
        data = np.zeros((len(train_files),num_features+1))
        for i in range(len(train_files)):
            rgb = read_tiff(filename=(path+train_files[i]))
            rgb = rgb_preprocess(rgb, verb=verb, exclude_bg=True, upper_lim=(0,  0,
                                                                         240))
            rh, gh, bh = rgb_histogram(rgb, verb=verb, omit=(0, 255))

            # Gathering features from histograms
            green_otsu = otsu_threshold(rgb[:, :, 1], verb=verb)
            blue_mode = calc_mode(bh)
            blue_median = calc_median(bh)
            blue_variance = calc_variance(bh)
            
            #features = np.append(green_otsu, [blue_mode,blue_median,blue_variance])
            features = np.append(green_otsu, [blue_mode])
            if ('dys' in train_files[i]):
                target = 1
            else:
                target = -1
            data[i] = np.append(target, [features])

        # Split data in to training and testing
        x_train, x_test, y_train, y_test = train_test_split(data[:,1:],data[:,0],
                                                            test_size = 0.3)
        # Train SVM
        svm = train_model(x_train,y_train,'basic_model.pkl')
        # Perform prediction on test set
        y_pred = class_predict(x_test,'basic_model.pkl')

        misclassification = len(np.nonzero(y_pred - y_test)) / len(y_test)
        accuracy = (1 - misclassification) * 100
        print ('Classification accuracy on test set = %f ' % accuracy)
        
        soft_predictions = svm.predict_proba(x_test)
        roc = calc_ROC(y_test, soft_predictions[:,1], True)
        auc = calc_AUC(y_test, soft_predictions[:,1])

        print ('AUC on test set = %f ' % auc)
        
    else:
        unknown_file = './test/ExampleAbnormalCervix.tif'
        rgb = read_tiff(filename= unknown_file)
        rgb = rgb_preprocess(rgb, verb=verb, exclude_bg=True, upper_lim=(0,  0,
                                                                         240))
        rh, gh, bh = rgb_histogram(rgb, verb=verb, omit=(0, 255))

        green_otsu = otsu_threshold(rgb[:, :, 1], verb=verb)
        blue_mode = calc_mode(bh)
        blue_median = calc_median(bh)
        blue_variance = calc_variance(bh)

        #features = np.append(green_otsu, [blue_mode,blue_median,blue_variance])
        features = np.append(green_otsu, [blue_mode])
        y_pred = class_predict(features.reshape(1,-1),'basic_model.pkl')
        if (y_pred == 1):
            print("SVM Classification Result = Dysplasia")
        else:
            print("SVM Classification Result = Healthy")

        msg = "G channel Otsu's threshold = %d" % green_otsu
        logging.info(msg)
        print(msg)

        msg = "B channel mode = %d" % blue_mode
        logging.info(msg)
        print(msg)
