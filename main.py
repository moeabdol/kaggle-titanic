import pandas as pd
from utils.load import load_data
from utils.preprocess import preprocess_data
from features.importance import use_most_important_features
from parameters.optimize import optimize_hyperparameters
import performance.learning_curves
from performance.learning_curves import plot_learning_curves
from performance.learning_curves import plot_ROC_curve
from prediction.predict import predict_survival

# set pandas print width
pd.options.display.width = 200

# load data
print("Loading Data...")
DIR = "./data"
train_df, test_df = load_data(DIR)

# preprocess, massage, scale, merge and clean data
print("Preprocessing Data...")
train_df, test_df = preprocess_data(train_df, test_df)

# use only most important features
print("Extracting Most Important Features...")
train_df, test_df = use_most_important_features(train_df, test_df)

# optimize hyperparameters
print("Optimizing Hyperparameters...")
optimize_hyperparameters(train_df)

# plot learning curves
print("Plot Learning Curves...")
plot_learning_curves(train_df)
plot_ROC_curve(train_df)

# predict survival
print("Predict Survival...")
predict_survival(train_df, test_df)
