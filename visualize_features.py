import os
import sys
import logging
import numpy as np
sys.path.insert(0, os.path.abspath('./src/'))


def parse_cli():
    """parse CLI

    :returns: args
    """
    import argparse as ap
    import os

    par = ap.ArgumentParser(description="2D Feature Visualization",
                            formatter_class=ap.ArgumentDefaultsHelpFormatter)

    par.add_argument("--train_dir",
                     dest="t_dir",
                     help="Full Path to Training Data",
                     default=os.getcwd() + '/TrainingData/')

    par.add_argument("--output_dir",
                     dest="out_dir",
                     help="Full Path to Output Directory",
                     default=os.getcwd() + '/outputs/')

    par.add_argument("--median_features",
                     dest="med_feats",
                     help="Color channels to extract median feature from <"
                          "rgb>",
                     default='')

    par.add_argument("--variance_features",
                     dest="var_feats",
                     help="Color channels to extract variance feature from <"
                          "rgb>",
                     default='')

    par.add_argument("--mode_features",
                     dest="mode_feats",
                     help="Color channels to extract mode feature from <rgb>",
                     default='')

    par.add_argument("--otsu_features",
                     dest="otsu_feats",
                     help="Color channels to extract otsu feature from <rgb>",
                     default='')

    par.add_argument("--yellow_feature",
                     dest="ypct_feat",
                     help="Use percent yellow pixel feature <True/False>",
                     type=str,
                     default='False')

    args = par.parse_args()
    args.ypct_feat = parse_bool(args.ypct_feat)

    return args


def parse_bool(inpar):
    """
    convert CLI True or False string into boolean

    :param inpar: input True or False string (True/False or T/F)
    :return: outpar (bool)
    """
    import sys
    msg = 'ERROR [parse_cli] CLI must be either True/False. ' \
          'Exiting script...'

    try:
        if inpar.lower() == 'true' or inpar.lower() == 't':
            outpar = True
        elif inpar.lower() == 'false' or inpar.lower() == 'f':
            outpar = False
        else:
            print(msg)
            sys.exit()

    except ValueError:
        print(msg)
        sys.exit()

    return outpar


if __name__ == "__main__":
    from preprocess import read_tiff, rgb_preprocess
    from feature_extraction import extract_features
    from classification_model import *
    from classification_model_metrics import *

    args = parse_cli()
    data_path = args.t_dir
    median_feats = args.med_feats
    variance_feats = args.var_feats
    mode_feats = args.mode_feats
    otsu_feats = args.otsu_feats
    ypct_feat = args.ypct_feat
    omit_pix = [0, 255]

    n_feat = len(median_feats+variance_feats+mode_feats+otsu_feats)

    if n_feat != 2:
        msg = 'ERROR script compatible with only 2D feature sets. Actual # ' \
              'features = %d' % n_feat
        print(msg)
        sys.exit()

    # threshold for glare filter
    b_thresh = 240

    train_files = os.listdir(data_path)
    n_train = len(train_files)

    target_array = np.zeros(n_train)
    feature_array = np.zeros((n_train, n_feat))

    for i in range(len(train_files)):

        msg = 'Extracting features from ' \
              + train_files[i] + ' (%d/%d)' % (i + 1, len(train_files))
        print(msg)

        rgb = read_tiff(filename=(data_path + train_files[i]))
        rgb = rgb_preprocess(rgb, verb=False, exclude_bg=True,
                             upper_lim=(0, 0, b_thresh))

        features = extract_features(rgb,
                                    median_ch=median_feats,
                                    variance_ch=variance_feats,
                                    mode_ch=mode_feats,
                                    otsu_ch=otsu_feats,
                                    pct_yellow=ypct_feat,
                                    omit=omit_pix,
                                    verb=False)

        feature_array[i, :] = features

        if 'dys' in train_files[i]:
            target_array[i] = 1
        else:
            target_array[i] = -1

    plot_features(feature_array, target_array, ['feature 1', 'feature 2'])
