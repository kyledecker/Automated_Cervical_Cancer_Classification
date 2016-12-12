# BME590 Final Project: Automated Cervical Cancer Diagnosis

## SUMMARY
This package is used to classify cervical cancer automatically using features extracted from POCkeT Colposcope images. The classification relies on a SVM model that is trained on a representative sample of dysplasia and healthy images.

## MODULES
* *main.py* - wrapper script used to either train and evaluate the SVM model or classify an unknown image 
* OTHER FILES HERE

## UNIT TESTING
Unit testing of core functions can performed by running *py.test* from the base directory

## RUN CODE
The automated cervical cancer diagnosis script can be run from the base directory using the following command:
```
python main.py
```

Ex. usage for _training_ SVM using default features and data set:
```
python main.py --train=True --train_dir=./TrainingData/ --verbose=True
```
Ex. usage for _classifying_ unknown image file:
```
python main.py --f=./ExampleAbnormalCervix.tif --verbose=True
```
For help and description of input arguments:
```
python main.py --help
```
**Be sure to set inputs for the image file you would like to classify, if you would like to perform training, the model filename, etc.** 

## OUTPUTS
All output files and figures are saved to a user-specified folder (default: ./outputs/). Intermediate figures can be saved by enabling verbose mode.