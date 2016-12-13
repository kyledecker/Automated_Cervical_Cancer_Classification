import os
import sys
import logging
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
    visualize 2D feature space for data set with known targets

    :param features: N x 2 array of features from N data sets
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
    from main import collect_training_data

    print('\nFEATURE SPACE VISUALIZATION')
    args = parse_cli_visualizer()
    data_path = args.t_dir
    outfile = args.out_file

    median_feats = args.med_feats
    variance_feats = args.var_feats
    mode_feats = args.mode_feats
    otsu_feats = args.otsu_feats
    pct_yellow = args.ypct_feat
    omit_pix = [0, 255]
    b_lim = 240

    feature_types = {'med': median_feats,
                     'var': variance_feats,
                     'mode': mode_feats,
                     'otsu': otsu_feats,
                     'ypct': pct_yellow}

    n_feat = len(feature_types['med'] + feature_types['var'] +
                 feature_types['mode'] + feature_types['otsu'])
    if feature_types['ypct']:
        n_feat += 1

    if n_feat != 2:
        msg = 'ERROR script compatible with only 2D feature sets. Actual # ' \
              'features = %d' % n_feat
        print(msg)
        sys.exit()

    x_train, x_test, y_train, y_test, feature_types, feature_labels = \
        collect_training_data(data_path, feature_types,
                              omit=omit_pix, b_cutoff=b_lim,
                              split_train_test_data=False,
                              verb=False)

    print('\nOUTPUTS')
    plot_features(x_train, y_train, feature_labels, outfile)
    print('\n')
