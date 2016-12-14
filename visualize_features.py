import os
import sys
sys.path.insert(0, os.path.abspath('./src/'))


if __name__ == "__main__":
    from feature_extraction import plot_features, collect_feature_data
    from parse_cli import parse_cli_visualizer

    print('\nFEATURE SPACE VISUALIZATION')
    args = parse_cli_visualizer()
    data_path = args.data_dir
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

    feature_array, target_array, feature_labels = \
        collect_feature_data(data_path, feature_types, omit=omit_pix,
                             b_cutoff=b_lim, verb=False)

    print('\nOUTPUTS')
    plot_features(feature_array, target_array, feature_labels, outfile)
