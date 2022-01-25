import numpy as np
from accessories import load_split_normalise_reweigh_dataset, lambda_grid_search, train_test_final_models, max_depth_grid_search, lambda_grid_search_cross_val, max_depth_grid_search_cross_val, train_test_final_models_RF

#########################################INPUT#########################################
#######################################################################################

# Do you wanna print info about the datasets?
print_dataset_sizes_on = True
# Do you wanna print all matrices produced for different λs?
print_cross_val_conf_matrices_on = True
# Select a dataset from the list of options: ["adult", "german", "compas"]
dataset = "compas"
# Select a dataset from the dict of options: ["adult": ['sex', 'race'], "german":['sex', 'age'], "compas":['sex', 'race']]
sensitive_feature = "sex"
# Select the classifier from the list of options: ["LogReg", "SVC", "RF"]
classifier_type = "LogReg"
# Do you wanna reweigh the training data for fairness?
reweigh_on = False
# Do you know the hyperparams of the fairest and most accurate models and just wanna perform against test data?
final_test_on = False

#######################################################################################
#######################################################################################

if final_test_on:
    if classifier_type == "RF":
        # model_hyperparams = {"most fair": 3, "most accurate": 8}
        model_hyperparams = {"best of both worlds": 4}
        for test_num in range(0, 5):
            (X_train, y_train, w_train, X_test, y_test, X_valid, y_valid, X_train_valid, y_train_valid, train, valid, test, train_valid,
             sens_feature_idx, privileged_groups, unprivileged_groups) = load_split_normalise_reweigh_dataset(dataset, sensitive_feature,
                                                                                                              [0.7, 0.15, 0.15], print_dataset_sizes_on,
                                                                                                              reweigh_on)

            data = (X_train, y_train, w_train, X_test, y_test, X_valid, y_valid, X_train_valid, y_train_valid, train, valid, test, train_valid)
            sensitive_metadata = (sens_feature_idx, privileged_groups, unprivileged_groups)

            models = train_test_final_models_RF(model_hyperparams, classifier_type, data, sensitive_metadata)
    else:
        model_hyperparams = {"best of both worlds": 10 ** 1}
        for test_num in range(0, 5):
            (X_train, y_train, w_train, X_test, y_test, X_valid, y_valid, X_train_valid, y_train_valid, train, valid, test, train_valid,
             sens_feature_idx, privileged_groups, unprivileged_groups) = load_split_normalise_reweigh_dataset(dataset, sensitive_feature,
                                                                                                              [0.7, 0.15, 0.15], print_dataset_sizes_on,
                                                                                                              reweigh_on)

            data = (X_train, y_train, w_train, X_test, y_test, X_valid, y_valid, X_train_valid, y_train_valid, train, valid, test, train_valid)
            sensitive_metadata = (sens_feature_idx, privileged_groups, unprivileged_groups)

            models = train_test_final_models(model_hyperparams, classifier_type, data, sensitive_metadata)

else:
    (X_train, y_train, w_train, X_test, y_test, X_valid, y_valid, X_train_valid, y_train_valid, train, valid, test, train_valid,
     sens_feature_idx, privileged_groups, unprivileged_groups) = load_split_normalise_reweigh_dataset(dataset, sensitive_feature,
                                                                                                      [0.7, 0.15, 0.15], print_dataset_sizes_on,
                                                                                                      reweigh_on)

    data = (X_train, y_train, w_train, X_test, y_test, X_valid, y_valid, X_train_valid, y_train_valid, train, valid, test, train_valid)
    sensitive_metadata = (sens_feature_idx, privileged_groups, unprivileged_groups)
    # Evaluate models created for different regularising hyperparameters and keep the most fair model (judged by average odds dif.) and the most accurate
    if classifier_type in ["LogReg", "SVC"]:
        lambda_values = [10 ** i for i in np.arange(-3, 8, 0.5)]
        model_hyperparams = lambda_grid_search(lambda_values, classifier_type, data, sensitive_metadata, print_cross_val_conf_matrices_on)
    else:
        max_depths = [i for i in range(2, 18)]
        model_hyperparams = max_depth_grid_search_cross_val(max_depths, classifier_type, data, sensitive_metadata,
                                                            print_cross_val_conf_matrices_on)



