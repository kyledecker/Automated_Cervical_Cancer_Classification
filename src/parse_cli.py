def parse_cli():
    """parse CLI

    :returns: args
    """
    import argparse as ap
    import os

    par = ap.ArgumentParser(description="Automated Cervical Cancer Diagnosis",
                            formatter_class=ap.ArgumentDefaultsHelpFormatter)

    par.add_argument("--f",
                     dest="f",
                     help="File Name of Image to be Classified",
                     default='./test/ExampleAbnormalCervix.tif')

    par.add_argument("--train_dir",
                     dest="t_dir",
                     help="Full Path to Training Data",
                     default=os.getcwd() + '/TrainingData/')

    par.add_argument("--output_dir",
                     dest="out_dir",
                     help="Full Path to Output Directory",
                     default=os.getcwd() + '/outputs/')

    par.add_argument("--verbose",
                     dest="v",
                     help="Display Image Histograms and Features <True/False>",
                     type=str,
                     default='False')

    par.add_argument("--train",
                     dest="t",
                     help="Train the Classifier <True/False>",
                     type=str,
                     default='False')

    par.add_argument("--model_filename",
                     dest="model",
                     help="Filename of the classification model",
                     default="dysplasia_svm_model.pkl")

    par.add_argument("--feature_set_filename",
                     dest="featset",
                     help="Filename of pkl containing training features",
                     default="dysplasia_svm_features.pkl")

    par.add_argument("--out_of_bag_testing",
                     dest="splitting",
                     help="Use out of bag samples for classification metrics "
                          "<True/False>",
                     type=str,
                     default='False')

    par.add_argument("--median_features",
                     dest="med_feats",
                     help="Color channels to extract median feature from <"
                          "rgb>",
                     default='')

    par.add_argument("--variance_features",
                     dest="var_feats",
                     help="Color channels to extract variance feature from <"
                          "rgb>",
                     default='rgb')

    par.add_argument("--mode_features",
                     dest="mode_feats",
                     help="Color channels to extract mode feature from <rgb>",
                     default='bg')

    par.add_argument("--otsu_features",
                     dest="otsu_feats",
                     help="Color channels to extract otsu feature from <rgb>",
                     default='')

    par.add_argument("--yellow_feature",
                     dest="ypct_feat",
                     help="Use percent yellow pixel feature <True/False>",
                     type=str,
                     default='False')

    par.add_argument("--log",
                     dest="l",
                     help="Logging Level <DEBUG, INFO, WARNING, ERROR>",
                     default='DEBUG')

    args = par.parse_args()

    args.v = parse_bool(args.v)
    args.t = parse_bool(args.t)
    args.ypct_feat = parse_bool(args.ypct_feat)
    args.splitting = parse_bool(args.splitting)

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
