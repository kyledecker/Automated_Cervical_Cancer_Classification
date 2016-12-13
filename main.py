import os
import sys
import logging
sys.path.insert(0, os.path.abspath('./src/'))


def collect_feature_data(filepath, feature_dict,
                          omit, b_cutoff=240,
                          verb=False, outdir='./outputs/'):
    """
    collect feature data from a specified directory

    :param filepath: directory containing all data
    :param feature_dict: dict of strings specifying color channel for features
    :param omit: pixel values to omit from calculation of features, ex [0, 255]
    :param b_cutoff: blue color channel cutoff for glare removal
    :param verb: verbose mode to save intermediate files and figures
    :param outdir: directory where output files are saved
    :return: feature_array, target_array, feature_labels
    """
    from preprocess import read_tiff, rgb_preprocess
    from feature_extraction import extract_features
    import numpy as np

    msg = 'Data location: %s' % filepath
    logging.info(msg)
    print(msg)

    msg = '\nSELECTED FEATURES:'
    logging.info(msg)
    print(msg)
    msg = 'Color channel median: %s' % feature_dict['med']
    logging.info(msg)
    print(msg)
    msg = 'Color channel variance: %s' % feature_dict['var']
    logging.info(msg)
    print(msg)
    msg = 'Color channel mode: %s' % feature_dict['mode']
    logging.info(msg)
    print(msg)
    msg = 'Color channel Otsu: %s' % feature_dict['otsu']
    logging.info(msg)
    print(msg)
    msg = 'Yellow pixel fraction: %s\n' % feature_dict['ypct']
    logging.info(msg)
    print(msg)

    n_feat = len(feature_dict['med'] + feature_dict['var'] +
                 feature_dict['mode'] + feature_dict['otsu'])
    if feature_dict['ypct']:
        n_feat += 1

    data_files = os.listdir(filepath)
    n_datasets = len(data_files)

    target_array = np.zeros(n_datasets)
    feature_array = np.zeros((n_datasets, n_feat))

    for i in range(len(data_files)):

        msg = 'Extracting features from ' \
              + data_files[i] + ' (%d/%d)' % (i + 1, len(data_files))
        logging.info(msg)
        print(msg)

        # directory to store outputs for training set
        feat_outdir = os.path.join(outdir, 'feature_data',
                                   os.path.splitext(data_files[i])[0])

        rgb = read_tiff(filename=(filepath + data_files[i]))
        rgb = rgb_preprocess(rgb, exclude_bg=True,
                             upper_lim=(0, 0, b_cutoff))

        features, l = extract_features(rgb,
                                       median_ch=feature_dict['med'],
                                       variance_ch=feature_dict['var'],
                                       mode_ch=feature_dict['mode'],
                                       otsu_ch=feature_dict['otsu'],
                                       pct_yellow=feature_dict['ypct'],
                                       omit=omit,
                                       verb=verb,
                                       outdir=feat_outdir)

        feature_array[i, :] = features

        if 'dys' in data_files[i]:
            target_array[i] = 1
        elif 'heal' in data_files[i]:
            target_array[i] = -1
        else:
            target_array[i] = None

        msg = 'Target label (1 dysplasia, -1 healthy): %d' % \
              target_array[i]
        logging.debug(msg)

    return feature_array, target_array, l


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
        while class_diff == False:
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


def classifier_metrics(x_test, y_test, y_pred, outdir='./outputs/'):
    """
    calculate and save classifier metrics to specified directory

    :param x_test: test data feature set (data set # x features)
    :param y_test: test data true targets
    :param y_pred: predicted targets
    :param outdir: directory where output files are saved
    :return: roc, auc, cm, accuracy, f1
    """
    from classification_model_metrics import calc_ROC, calc_AUC, \
        gen_confusion_matrix, calc_accuracy, calc_f1_score

    msg = '\nOUTPUTS'
    logging.info(msg)
    print(msg)

    soft_predictions = svm.predict_proba(x_test)

    outfile = os.path.join(outdir, 'roc.png')
    roc = calc_ROC(y_test, soft_predictions[:, 1], True, outfile=outfile)
    auc = calc_AUC(y_test, soft_predictions[:, 1])

    outfile = os.path.join(outdir, 'confusionmat.png')
    cm = gen_confusion_matrix(y_test, y_pred, ('Healthy', 'Dysp.'),
                              verb=True, outfile=outfile)

    msg = '\n***** RESULTS *****'
    logging.info(msg)
    print(msg)

    accuracy = calc_accuracy(y_test, y_pred)
    msg = 'Classification accuracy = %.1f ' % accuracy
    logging.info(msg)
    print(msg)

    f1 = calc_f1_score(y_test, y_pred)
    msg = 'F1-score on test set = %.1f ' % f1
    logging.info(msg)
    print(msg)

    msg = 'AUC on test set = %.1f ' % auc
    logging.info(msg)
    print(msg)

    msg = '*Additional results in outputs folder.' \
          '\n*******************\n'
    logging.info(msg)
    print(msg)

    return roc, auc, cm, accuracy, f1


if __name__ == "__main__":
    from classification_model import *
    from parse_cli import parse_cli
    import pickle

    # gather general CLI
    args = parse_cli()
    verb = args.v
    train = args.t
    model_filename = args.model
    featset_filename = args.featset
    outdir = args.out_dir
    log_level = args.l

    # configure logging
    logging.basicConfig(filename="log.txt", level=log_level,
                        format='%(asctime)s - %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    logging.debug('Running Automated Cervical Cancer Diagnosis.')

    # pixels to omit from feature extraction
    omit_pix = [0, 255]
    # threshold for glare filter
    b_lim = 240

    if train:

        split_train_test = args.splitting
        data_path = args.t_dir
        median_feats = args.med_feats
        variance_feats = args.var_feats
        mode_feats = args.mode_feats
        otsu_feats = args.otsu_feats
        pct_yellow = args.ypct_feat

        feature_types = {'med': median_feats,
                         'var': variance_feats,
                         'mode': mode_feats,
                         'otsu': otsu_feats,
                         'ypct': pct_yellow}

        msg = '\nTRAINING'
        logging.info(msg)
        print(msg)

        pickle.dump(feature_types, open(featset_filename, 'wb'))

        msg = 'Training feature set saved: %s' % featset_filename
        logging.info(msg)
        print(msg)

        # perform feature extraction and collect training data
        feature_array, target_array, feature_labels = \
            collect_feature_data(data_path, feature_types,
                                 omit=omit_pix, b_cutoff=b_lim,
                                 verb=verb, outdir=outdir)

        x_train, x_test, y_train, y_test = split_data(feature_array,
                                                      target_array,
                                                      split_train_test)

        # Train SVM
        msg = 'Training SVM classifier...'
        logging.info(msg)
        print(msg)
        svm = train_model(x_train, y_train, model_filename)

        # Perform prediction on test set
        msg = 'Performing prediction on test data and generating metrics...'
        logging.info(msg)
        print(msg)
        y_pred = class_predict(x_test, model_filename)

        # Calculate and save output metrics
        classifier_metrics(x_test, y_test, y_pred, outdir=outdir)
        
    else:
        from preprocess import read_tiff, rgb_preprocess
        from feature_extraction import extract_features
        from feature_extraction import calc_pct_yellow

        # gather prediction specific CLI
        unknown_file = args.f

        msg = '\nPREDICTION'
        logging.info(msg)
        print(msg)

        msg = 'Target prediction file: %s' % model_filename
        logging.info(msg)
        print(msg)

        # directory for prediction outputs
        pred_outdir = os.path.join(outdir, 'prediction/')

        try:
            msg = 'Feature set: %s' % featset_filename
            logging.info(msg)
            print(msg)
            feature_types = pickle.load(open(featset_filename, 'rb'))
        except FileNotFoundError:
            msg = 'Error loading feature info file: %s \n' \
                'Ensure that you have previously trained the model by running ' \
                'main.py with train=True \n' \
                'Exiting script...' % featset_filename
            print(msg)
            logging.error(msg)
            sys.exit()

        rgb = read_tiff(filename=unknown_file)
        rgb = rgb_preprocess(rgb, exclude_bg=True,
                             upper_lim=(0, 0, b_lim))

        features, l = extract_features(rgb,
                                       median_ch=feature_types['med'],
                                       variance_ch=feature_types['var'],
                                       mode_ch=feature_types['mode'],
                                       otsu_ch=feature_types['otsu'],
                                       pct_yellow=feature_types['ypct'],
                                       omit=omit_pix,
                                       verb=verb,
                                       outdir=pred_outdir)

        y_pred = class_predict(features.reshape(1, -1), model_filename)

        msg = '\nOUTPUTS'
        logging.info(msg)
        print(msg)

        if y_pred == 1:
            outfile = os.path.join(pred_outdir, 'labeled_lesion.png')
            pct_les = calc_pct_yellow(rgb, verb=True, outfile=outfile)

            msg = '\n***** RESULTS *****'
            logging.info(msg)
            print(msg)

            msg = "SVM Classification Result = Dysplasia"
            logging.info(msg)
            print(msg)

            msg = "Percent Lesion = %.1f %%" % pct_les
            logging.info(msg)
            print(msg)

        else:
            msg = '\n***** RESULTS *****'
            logging.info(msg)
            print(msg)

            msg = "SVM Classification Result = Healthy"
            logging.info(msg)
            print(msg)

        msg = '*Additional results in outputs folder.' \
              '\n*******************\n'
        logging.info(msg)
        print(msg)
