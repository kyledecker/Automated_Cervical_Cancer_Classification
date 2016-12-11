import os
import sys
import logging
import numpy as np
sys.path.insert(0, os.path.abspath('./src/'))


if __name__ == "__main__":
    from preprocess import read_tiff, rgb_preprocess
    from feature_extraction import extract_features
    from classification_model import *
    from classification_model_metrics import *

    verb = False

    data_path = os.getcwd() + '/TrainingData/'
    median_feats = 'rgb'
    variance_feats = 'rgb'
    mode_feats = 'rgb'
    otsu_feats = 'rgb'
    omit_pix = [0, 255]

    n_feat = len(median_feats+variance_feats+mode_feats+otsu_feats)

    # threshold for glare filter
    b_thresh = 240

    train_files = os.listdir(data_path)
    n_train = len(train_files)

    target_array = np.zeros(n_train)
    feature_array = np.zeros((n_train, n_feat))

    for i in range(len(train_files)):
        rgb = read_tiff(filename=(data_path + train_files[i]))
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

    labels_array = ['R median', 'G median', 'B median', 'R variance',
                    'G variance', 'B variance', 'R mode', 'G mode',
                    'B mode', 'R threshold', 'G threshold', 'B threshold']

    for ii in range(0, len(labels_array)-1):
        plot_features(feature_array[:, [ii, ii+1]], target_array,
                      [labels_array[ii], labels_array[ii+1]])



