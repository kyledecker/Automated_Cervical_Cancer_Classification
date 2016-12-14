import argparse as ap
import os


def parse_cli_main():
    """parse CLI for main function

    :returns: args
    """
    par = ap.ArgumentParser(description="Automated Cervical Cancer Diagnosis",
                            formatter_class=ap.ArgumentDefaultsHelpFormatter)

    par.add_argument("--predict",
                     dest="predict",
                     help="File or Directory of Files to be Classified",
                     default='./ExampleAbnormalCervix.tif')

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
                     help="Display Image Histograms and Features",
                     action='store_true')

    par.add_argument("--train",
                     dest="t",
                     help="Train the Classifier",
                     action='store_true')

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
                     help="Use out of bag samples for classification metrics",
                     action='store_true')

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
                     help="Use percent yellow pixel feature",
                     action='store_true')

    par.add_argument("--log",
                     dest="l",
                     help="Logging Level <DEBUG, INFO, WARNING, ERROR>",
                     default='DEBUG')

    args = par.parse_args()

    return args


def parse_cli_visualizer():
    """parse CLI for feature visualization tool

    :returns: args
    """
    par = ap.ArgumentParser(description="2D Feature Visualization",
                            formatter_class=ap.ArgumentDefaultsHelpFormatter)

    par.add_argument("--data_dir",
                     dest="data_dir",
                     help="Full Path to Data",
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
                     action='store_true')

    args = par.parse_args()

    return args
