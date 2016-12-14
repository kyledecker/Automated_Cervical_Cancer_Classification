# BME590 Final Project: Automated Cervical Cancer Diagnosis

## SUMMARY
This package is used to classify cervical cancer automatically using features extracted from POCkeT Colposcope images. The classification relies on a SVM model that is trained on a representative sample of dysplasia and healthy images.

## MODULES
* *main.py* - wrapper script used to either train and evaluate the SVM model or classify an unknown image 
* *parse_cli.py* - parses command line inputs to run main function
* *preprocess.py* - reads in image files and prepares them for feature extraction
* *feature_extraction.py* - extracts features used for classification
* *classification_model.py* - performs SVM training and prediction
* *classification_model_metrics.py* - computes various metrics to evaluate classifier performance
* *accessory.py* - performs miscellaneous data manipulation, saving, and display 

## UNIT TESTING
Unit testing of core functions can performed by running *py.test* from the base directory

## RUN CODE
The automated cervical cancer diagnosis script can be run from the base directory using the following command:
```
python main.py
```

Ex. usage for _training_ SVM using default features and data set:
```
python main.py --train --train_dir=./TrainingData/ --verbose
```
Ex. usage for _classifying_ unknown image file:
```
python main.py --f=./ExampleAbnormalCervix.tif --verbose
```
Ex. usage for _classifying_ directory of unknown image files:
```
python main.py --f=./dysplasia/ --verbose
```
For help and description of input arguments:
```
python main.py --help
```
**Be sure to set inputs for the image file you would like to classify, if you would like to perform training, the model filename, etc.** 

## OUTPUTS
All output files and figures are saved to a user-specified folder (default: ./outputs/). Intermediate files can be saved by enabling verbose mode.

## EXTRAS
A visualization tool is available for plotting the 2D feature space of a data set with known targets. Use CLI to specify the desired features to plot and location of training set. 

Ex. usage for plotting R median and B variance feature space:
```
python visualize_features.py --data_dir=./TrainingData/ --med=r --var=b
```