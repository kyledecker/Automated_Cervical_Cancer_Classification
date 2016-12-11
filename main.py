import os
import sys
import logging
import numpy as np
sys.path.insert(0, os.path.abspath('./src/'))


if __name__ == "__main__":
    from preprocess import read_tiff, rgb_preprocess
    from feature_extraction import extract_features, calc_pct_yellow
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
    pct_yellow = args.ypct_feat

    omit_pix = [0, 255]

    n_feat = len(median_feats+variance_feats+mode_feats+otsu_feats)
    if pct_yellow:
        n_feat += 1

    # threshold for glare filter
    b_thresh = 240

    if train:
        train_files = os.listdir(data_path)
        n_train = len(train_files)

        target_array = np.zeros(n_train)
        feature_array = np.zeros((n_train, n_feat))

        for i in range(len(train_files)):

            msg = 'Extracting features from ' \
                  + train_files[i] + ' [%d/%d]' % (i+1, len(train_files))
            logging.info(msg)
            print(msg)

            rgb = read_tiff(filename=(data_path+train_files[i]))
            rgb = rgb_preprocess(rgb, verb=verb, exclude_bg=True,
                                 upper_lim=(0, 0, b_thresh))

            features = extract_features(rgb,
                                        median_ch=median_feats,
                                        variance_ch=variance_feats,
                                        mode_ch=mode_feats,
                                        otsu_ch=otsu_feats,
                                        pct_yellow=pct_yellow,
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
        msg = 'Classification accuracy on test set = %.1f ' % accuracy
        logging.info(msg)
        print(msg)

        f1 = calc_f1_score(y_test, y_pred)
        msg = 'F1-score on test set = %.1f ' % f1
        logging.info(msg)
        print(msg)
        
        soft_predictions = svm.predict_proba(x_test)
        roc = calc_ROC(y_test, soft_predictions[:, 1], True)
        auc = calc_AUC(y_test, soft_predictions[:, 1])

        msg = 'AUC on test set = %.1f ' % auc
        logging.info(msg)
        print(msg)

        gen_confusion_matrix(y_test, y_pred, ('Dysplasia', 'Healthy'),
                             verb=True)
        
    else:
        rgb = read_tiff(filename=unknown_file)
        rgb = rgb_preprocess(rgb, verb=verb, exclude_bg=True,
                             upper_lim=(0, 0, b_thresh))

        features = extract_features(rgb,
                                    median_ch=median_feats,
                                    variance_ch=variance_feats,
                                    mode_ch=mode_feats,
                                    otsu_ch=otsu_feats,
                                    pct_yellow=pct_yellow,
                                    omit=[0, 255],
                                    verb=verb)

        y_pred = class_predict(features.reshape(1, -1), model_filename)
        pct_disease = calc_pct_yellow(rgb)

        if y_pred == 1:
            msg = "SVM Classification Result = Dysplasia"
            logging.info(msg)
            print(msg)

            msg = "Percent Diseased = %.1f" % pct_disease
            logging.info(msg)
            print(msg)
        else:
            msg = "SVM Classification Result = Healthy"
            logging.info(msg)
            print(msg)
