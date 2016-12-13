import os
import sys
import logging
import numpy as np
sys.path.insert(0, os.path.abspath('./src/'))


def parse_cli_visualizer():
    """parse CLI

    :returns: args
    """
    from parse_cli import parse_bool
    import argparse as ap
    import os

    par = ap.ArgumentParser(description="2D Feature Visualization",
                            formatter_class=ap.ArgumentDefaultsHelpFormatter)

    par.add_argument("--train_dir",
                     dest="t_dir",
                     help="Full Path to Training Data",
                     default=os.getcwd() + '/TrainingData/')

    par.add_argument("--output_file",
                     dest="out_file",
                     help="Full Path to Output File",
                     default=os.getcwd() + '/outputs/feature_plot.png')

    par.add_argument("--med",
                     dest="med_feats",
                     help="Color channels to extract median feature from <"
                          "rgb>",
                     default='')

    par.add_argument("--var",
                     dest="var_feats",
                     help="Color channels to extract variance feature from <"
                          "rgb>",
                     default='')

    par.add_argument("--mode",
                     dest="mode_feats",
                     help="Color channels to extract mode feature from <rgb>",
                     default='')

    par.add_argument("--otsu",
                     dest="otsu_feats",
                     help="Color channels to extract otsu feature from <rgb>",
                     default='')

    par.add_argument("--y",
                     dest="ypct_feat",
                     help="Use percent yellow pixel feature <True/False>",
                     type=str,
                     default='False')

    args = par.parse_args()
    args.ypct_feat = parse_bool(args.ypct_feat)

    return args


def plot_features(features, targets, labels, outfile='feat_plot.png'):
    """
    visualize 2D features for samples from 2 known classes

    :param features: N x 2 sets of feature values
    :param targets: target labels corresponding to each set of features
    :param labels: labels for each feature
    :param outfile: save location of output feature plot
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from accessory import create_dir

    target_types = np.unique(targets)
    if len(target_types) > 2:
        msg = 'ERROR [plot_features] Function only compatible with 2 targets.'
        logging.error(msg)
        print(msg)
        sys.exit()

    if np.shape(features)[1] > 2:
        msg = 'ERROR [plot_features] Function only compatible with 2 features.'
        logging.error(msg)
        print(msg)
        sys.exit()

    if np.shape(features)[0] != len(targets):
        msg = 'ERROR [plot_features] Mismatch between number of target ' \
              'labels and feature sets.'
        logging.error(msg)
        print(msg)
        sys.exit()

    features0 = features[targets == target_types[0], :]
    features1 = features[targets == target_types[1], :]

    h0 = plt.scatter(features0[:, 0], features0[:, 1], marker='o', c='red',
                     label=target_types[0])
    h1 = plt.scatter(features1[:, 0], features1[:, 1], marker='o',
                     c='blue',
                     label=target_types[1])
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend(handles=[h0, h1], loc=4)
    plt.grid(True)
    plt.axis('tight')

    create_dir(outfile)
    msg = '[plot_features] Feature space plot saved: %s' % outfile
    print(msg)
    plt.savefig(outfile)


if __name__ == "__main__":
    from preprocess import read_tiff, rgb_preprocess
    from feature_extraction import extract_features
    from classification_model import *
    from classification_model_metrics import *

    print('\nFEATURE SPACE VISUALIZATION')
    args = parse_cli_visualizer()
    data_path = args.t_dir
    outfile = args.out_file

    median_feats = args.med_feats
    variance_feats = args.var_feats
    mode_feats = args.mode_feats
    otsu_feats = args.otsu_feats
    ypct_feat = args.ypct_feat
    omit_pix = [0, 255]

    n_feat = len(median_feats+variance_feats+mode_feats+otsu_feats)
    if ypct_feat:
        n_feat += 1

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

        features, l = extract_features(rgb,
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

    print('\nOUTPUTS')
    plot_features(feature_array, target_array, l, outfile)
    print('\n')
