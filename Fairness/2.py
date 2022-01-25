import numpy as np
from aif360.datasets import AdultDataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult

from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing.reweighing import Reweighing

import pdb

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.model_selection import *

from accessories2 import load_split_normalise_reweight_dataset, lambda_grid_search, train_test_final_models, max_depth_grid_search

np.random.seed(0)

#########################################INPUT#########################################
#######################################################################################

# Do you wanna print info about the datasets?
print_dataset_sizes_on = True
# Do you wanna print all matrices produced for different λs?
print_cross_val_conf_matrices_on = True
# Select a dataset from the list of options: ["adult", "german", "compas"]
dataset = "adult"
# Select a dataset from the dict of options: ["adult": ['sex', 'race'], "german":['sex', 'age'], "compas":['sex', 'race']]
sensitive_feature = "sex"
# Select the classifier from the list of options: ["LogReg", "SVC"]
classifier_type = "RF"

#######################################################################################
#######################################################################################


(X_train, y_train, w_train, X_test, y_test, X_valid, y_valid, X_train_valid, y_train_valid,
train, train_transf, valid, test, sens_feature_idx, privileged_groups, unprivileged_groups) = load_split_normalise_reweight_dataset(dataset, sensitive_feature,
                                                                                                                                    [0.7, 0.15, 0.15],
                                                                                                                                    print_dataset_sizes_on)

data = (X_train, y_train, w_train, X_test, y_test, X_valid, y_valid, X_train_valid, y_train_valid, train, valid, test)
sensitive_metadata = (sens_feature_idx, privileged_groups, unprivileged_groups)

# Evaluate models created for different regularising hyperparameters and keep the most fair model (judged by average odds dif.) and the most accurate
if classifier_type in ["LogReg", "SVC"]:
    lambda_values = [10 ** i for i in np.arange(-3, 8, 0.5)]
    model_hyperparams = lambda_grid_search(lambda_values, classifier_type, data, sensitive_metadata, print_cross_val_conf_matrices_on)
else:
    max_depths = [i for i in range(2, 12)]
    model_hyperparams = max_depth_grid_search(max_depths, classifier_type, data, sensitive_metadata,
                                              print_cross_val_conf_matrices_on)

# models = train_test_final_models(model_hyperparams, classifier_type, data, sensitive_metadata)
