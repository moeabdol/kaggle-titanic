Introduction
--
My solution to the Kaggle Titanic competition. Achieving accuracy score of 78% (0.77512).  Note: running the code may last hours. It took around 2 hours of execution time on an early 2014 MacBook Pro 2.3Ghz 8 core machine.

Instructions
--
$ python main.py

Dependencies
--
python 2.7.9

pandas 0.15.2

sklearn 0.15.2

matplotlib 1.4.2

Project Files
--
README.md                   This readme file describing the project.

test.csv                    Testing dataset.

train.csv                   Training dataset.

main.py                     Starting execution point of this project.

age.py                      Script handeling the age feature.

embarked.py                 Script handeling the embarked feature.

importance.py               Script handeling extracting feature importances.

interaction_features.py     Script handeling the creation of interaction features.

name.py                     Script handeling the name feature.

scale.py                    Script handeling the scaling of features.

sex.py                      Script handeling the sex feature.

featrue_importances.png     Figure demonstrating feature importances.

learning_curves.png         Figure demonstrating the training and testing learning accuracy curves.

roc_curve.png               Figure demonstrating the ROC curve.

optimize.py                 Script handeling the optemization of the classifier's hyperparameters.

learning_curves.py          Script handeling the creation and ploting of learning curves.

predict.py                  Script handeling the survival prediction of the testing set.

result.csv                  The output prediction set.

best_params.csv             Intermediate result of the optimal hyperparameters.

load.py                     Script handeling the loading of training and testing datasets.

preprocess.py               Script handeling the preprocessing and cleaning of datasets.
