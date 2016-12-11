import os
import sys
import logging
import numpy as np
sys.path.insert(0, os.path.abspath('./src/'))


if __name__ == "__main__":
    from preprocess import read_tiff, rgb_histogram, rgb_preprocess
    from feature_extraction import extract_features
    from classification_model import *
    from sklearn.model_selection import train_test_split
    from classification_model_metrics import *
    from parse_cli import parse_cli

    # gather argparse inputs
    args = parse_cli()
    verb = args.v
    train = args.t
    split_train_test = args.splitting
    data_path = args.t_dir
    median_feats = args.med_feats
    variance_feats = args.var_feats
    mode_feats = args.mode_feats
    otsu_feats = args.otsu_feats
    unknown_file = args.f
    model_filename = args.model

    omit_pix = [0, 255]

    n_feat = len(median_feats+variance_feats+mode_feats+otsu_feats)

    # threshold for glare filter
    b_thresh = 240

    if train:
        train_files = os.listdir(data_path)
        n_train = len(train_files)

        target_array = np.zeros(n_train)
        feature_array = np.zeros((n_train, n_feat))

        for i in range(len(train_files)):
            rgb = read_tiff(filename=(data_path+train_files[i]))
            rgb = rgb_preprocess(rgb, verb=verb, exclude_bg=True,
                                 upper_lim=(0, 0, b_thresh))

            features = extract_features(rgb,
                                        median_ch=median_feats,
                                        variance_ch=variance_feats,
                                        mode_ch=mode_feats,
                                        otsu_ch=otsu_feats,
                                        omit=omit_pix,
                                        verb=verb)

            feature_array[i, :] = features

            if 'dys' in train_files[i]:
                target_array[i] = 1
            else:
                target_array[i] = -1

        if split_train_test:
            # Split data in to training and testing (best practice)
            x_train, x_test, y_train, y_test \
                = train_test_split(feature_array, target_array, test_size=0.3)
        else:
            # Use same data to train and test SVM
            x_train = feature_array
            y_train = target_array
            x_test = feature_array
            y_test = target_array

        # Train SVM
        svm = train_model(x_train, y_train, model_filename)

        # Perform prediction on test set
        y_pred = class_predict(x_test, model_filename)

        accuracy = calc_accuracy(y_test, y_pred)
        print('Classification accuracy on test set = %1f ' % accuracy)

        f1 = calc_f1_score(y_test, y_pred)
        print('F1-score on test set = %1f ' % f1)
        
        soft_predictions = svm.predict_proba(x_test)
        roc = calc_ROC(y_test, soft_predictions[:, 1], True)
        auc = calc_AUC(y_test, soft_predictions[:, 1])

        gen_confusion_matrix(y_test, y_pred, ('Dysplasia', 'Healthy'),
                             verb=True)

        print('AUC on test set = %f ' % auc)
        
    else:
        rgb = read_tiff(filename=unknown_file)
        rgb = rgb_preprocess(rgb, verb=verb, exclude_bg=True,
                             upper_lim=(0, 0, b_thresh))

        features = extract_features(rgb,
                                    median_ch=median_feats,
                                    variance_ch=variance_feats,
                                    mode_ch=mode_feats,
                                    otsu_ch=otsu_feats,
                                    omit=[0, 255])

        y_pred = class_predict(features.reshape(1, -1), model_filename)

        if y_pred == 1:
            print("SVM Classification Result = Dysplasia")
        else:
            print("SVM Classification Result = Healthy")
