Instructions
------------
$ python main.py
-------------------------------------------------------------------------------
Dependencies
------------
python 2.7.9
pandas 0.15.2
sklearn 0.15.2
matplotlib 1.4.2
-------------------------------------------------------------------------------
Project Structure
-----------------
titanic/
├── README.txt                      This readme file describing the project.
├── data
│   ├── test.csv                    Testing dataset.
│   └── train.csv                   Training dataset.
├── features
│   ├── __init__.py
│   ├── age.py                      Script handeling the age feature.
│   ├── embarked.py                 Script handeling the embarked feature.
│   ├── importance.py               Script handeling extracting feature importances.
│   ├── interaction_features.py     Script handeling the creation of interaction features.
│   ├── name.py                     Script handeling the name feature.
│   ├── scale.py                    Script handeling the scaling of features.
│   └── sex.py                      Script handeling the sex feature.
├── figures
│   ├── featrue_importances.png     Figure demonstrating feature importances.
│   ├── learning_curves.png         Figure demonstrating the training and testing learning accuracy curves.
│   └── roc_curve.png               Figure demonstrating the ROC curve.
├── main.py                         Starting execution point of this project.
├── parameters
│   ├── __init__.py
│   └── optimize.py                 Script handeling the optemization of the classifier's hyperparameters.
├── performance
│   ├── __init__.py
│   └── learning_curves.py          Script hendeling the creation and ploting of learning curves.
├── prediction
│   ├── __init__.py
│   └── predict.py                  Script handeling the survival prediction of the testing set.
├── result
│   └── result.csv                  The output prediction set.
├── temp
│   └── best_params.csv             Intermediate result of the optimal hyperparameters.
└── utils
    ├── __init__.py
    ├── load.py                     Script handeling the loading of training and testing datasets.
    └── preprocess.py               Script handeling the preprocessing and cleaning of datasets.
-------------------------------------------------------------------------------
