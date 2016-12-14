import os
import sys
import logging
sys.path.insert(0, os.path.abspath('./src/'))


if __name__ == "__main__":
    from classification_model import *
    from classification_model_metrics import classifier_metrics, \
        prediction_metrics
    from feature_extraction import collect_feature_data
    from parse_cli import parse_cli_main
    import pickle

    # gather general CLI
    args = parse_cli_main()
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

    # SVM TRAINING
    if train:
        split_train_test = args.splitting
        data_path = args.t_dir
        median_feats = args.med_feats
        variance_feats = args.var_feats
        mode_feats = args.mode_feats
        otsu_feats = args.otsu_feats
        pct_yellow = args.ypct_feat
        k_folds = args.kfolds

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
        classifier_metrics(svm, x_test, y_test, y_pred,
                           feature_array, target_array, k_folds, outdir=outdir)

    # SVM PREDICTION
    else:
        # gather prediction specific CLI
        unknown_data = args.predict

        msg = '\nPREDICTION'
        logging.info(msg)
        print(msg)

        msg = 'Target prediction file(s): %s' % unknown_data
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
                  'Ensure that you have previously trained the model by ' \
                  'running: python main.py --train' % featset_filename
            print(msg)
            logging.error(msg)
            sys.exit()

        feature_array, target_array, feature_labels = \
            collect_feature_data(unknown_data, feature_types,
                                 omit=omit_pix, b_cutoff=b_lim,
                                 verb=verb, outdir=outdir)
        # perform prediction
        y_pred = class_predict(feature_array, model_filename)

        # output prediction metrics and save lesion-labeled images
        prediction_metrics(unknown_data, y_pred, target_array, b_cutoff=b_lim,
                           outdir=pred_outdir)
