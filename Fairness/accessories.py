from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing.reweighing import Reweighing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

def plot(x, x_name, y_left, y_left_name, y_right, y_right_name, title, classifier_type):

    fig, ax1 = plt.subplots(figsize=(10,7))
    ax1.set_title(title, fontsize=16, fontweight='bold')
    if classifier_type != "RF":
        ax1.set_xscale("log")
    ax1.plot(x, y_left)
    ax1.set_xlabel(x_name, fontsize=16, fontweight='bold')
    ax1.set_ylabel(y_left_name, color='b', fontsize=16, fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)
    #ax1.set_ylim(0.5, 0.8)

    ax2 = ax1.twinx()
    if classifier_type != "RF":
        ax2.set_xscale("log")
    ax2.plot(x, y_right, color='r')
    ax2.set_ylabel(y_right_name, color='r', fontsize=16, fontweight='bold')
    #ax2.set_ylim(-0.3, 0.2)

    plt.show()
    pass


def load_split_normalise_reweigh_dataset(dataset: str, sensitive_feature: str,
                                         train_valid_test_proportions: list, print_dataset_sizes_on: bool,
                                         reweigh_on: bool) -> tuple:

    # Load the dataset and split it into train, test and validation
    if dataset == "adult":
        ad = load_preproc_data_adult([sensitive_feature])
    elif dataset == "german":
        ad = load_preproc_data_german([sensitive_feature])
    elif dataset == "compas":
        ad = load_preproc_data_compas([sensitive_feature])
    else:
        raise ValueError("Invalid dataset name!")

    # Set sex to be the sensitive feature
    privileged_groups = [{sensitive_feature: ad.favorable_label}]
    unprivileged_groups = [{sensitive_feature: ad.unfavorable_label}]

    train_valid, test = ad.split([train_valid_test_proportions[0] + train_valid_test_proportions[1]], shuffle=True)
    train, valid = train_valid.split([float(train_valid_test_proportions[0] / (1 - train_valid_test_proportions[2]))])

    # Finding the index of the sensitive feature "sex"
    sens_feature_idx = ad.feature_names.index(sensitive_feature)

    # Reweigh the training dataset
    if reweigh_on:
        RW = Reweighing(unprivileged_groups=unprivileged_groups,
                        privileged_groups=privileged_groups)
        RW.fit(train)
        train_transf = RW.transform(train)

    # Normalise train, test and valid datasets
    scaler = StandardScaler()
    if reweigh_on:
        scaler = scaler.fit(train_transf.features)
    else:
        scaler = scaler.fit(train.features)

    if reweigh_on:
        X_train = scaler.transform(train_transf.features)
        y_train = train_transf.labels.ravel()
        w_train = train_transf.instance_weights.ravel()
    else:
        X_train = scaler.transform(train.features)
        y_train = train.labels.ravel()
        w_train = train.instance_weights.ravel()
    X_test = scaler.transform(test.features)
    y_test = test.labels.ravel()
    X_valid = scaler.transform(valid.features)
    y_valid = valid.labels.ravel()
    X_train_valid = scaler.transform(train_valid.features)
    y_train_valid = train_valid.labels.ravel()

    # Print info about the datasets
    if print_dataset_sizes_on:
        print(100 * "-")
        print("X_train")
        print("type:", type(X_train))
        print("shape:", X_train.shape)
        print(100 * "-")
        print("y_train")
        print("type:", type(y_train))
        print("shape:", y_train.shape)
        print(100 * "-")
        print("X_valid")
        print("type:", type(X_valid))
        print("shape:", X_valid.shape)
        print(100 * "-")
        print("y_valid")
        print("type:", type(y_valid))
        print("shape:", y_valid.shape)
        print(100 * "-")
        print("X_test")
        print("type:", type(X_test))
        print("shape:", X_test.shape)
        print(100 * "-")
        print("y_test")
        print("type:", type(y_test))
        print("shape:", y_test.shape)
        print(100 * "-")
        print("X_train_valid")
        print("type:", type(X_train_valid))
        print("shape:", X_train_valid.shape)
        print(100 * "-")
        print("y_train_valid")
        print("type:", type(y_train_valid))
        print("shape:", y_train_valid.shape)
        print(100 * "-")
        print("Privileged Group Label =", test.favorable_label, "(Unprivileged Group Label = {})".format(test.unfavorable_label))
        print("Protected Attributes =", test.protected_attribute_names)
        print("Features =", test.feature_names)
        print(100 * "-")

    return (X_train, y_train, w_train, X_test, y_test, X_valid, y_valid, X_train_valid, y_train_valid, train, valid, test, train_valid,
            sens_feature_idx, privileged_groups, unprivileged_groups)


def confusion_matrix_per_class(X: np.array, y: np.array, y_pred: np.array, sens_feature_idx: int):

    priv_predictions = []
    priv_truths = []
    unpriv_predictions = []
    unpriv_truths = []

    for idx, prediction in enumerate(y_pred):
        if X[idx][sens_feature_idx] > 0:
            priv_truths.append(y[idx])
            priv_predictions.append(prediction)
        else:
            unpriv_truths.append(y[idx])
            unpriv_predictions.append(prediction)

    confusion_matrix_priv = confusion_matrix(priv_truths, priv_predictions)
    confusion_matrix_unpriv = confusion_matrix(unpriv_truths, unpriv_predictions)

    return confusion_matrix_priv, confusion_matrix_unpriv


def lambda_grid_search(lambda_values: list, classifier_type: str, data: tuple, sensitive_metadata: tuple,
                       print_cross_val_conf_matrices_on: bool):

    X_train, y_train, w_train, X_test, y_test, X_valid, y_valid, X_train_valid, y_train_valid, train, valid, test, train_valid = data
    sens_feature_idx, privileged_groups, unprivileged_groups = sensitive_metadata

    models_for_report = dict()
    average_odds_dif_most_fair = -float('inf')
    accuracy_most_accurate = 0

    lambdas = []
    accuracies = []
    avg_odds_difs = []

    # Train many models for different hyperparameter value, print the accuracy and fairness results of each one of them,
    # (if print_on), and note the hyperparameters of the most accurate and most fair models
    for lam in lambda_values:
        C = 1 / lam

        if classifier_type == "LogReg":
            model = LogisticRegression(solver='liblinear', random_state=1, C=C)
        elif classifier_type == "SVC":
            model = SVC(C=C)
        else:
            raise ValueError("Invalid classifier type!")

        model.fit(X_train, y_train, sample_weight=w_train)
        y_valid_pred = model.predict(X_valid)

        confusion_matrix_total = confusion_matrix(y_valid, y_valid_pred)
        accuracy_total = accuracy_score(y_valid, y_valid_pred)
        confusion_matrix_priv, confusion_matrix_unpriv = confusion_matrix_per_class(X_valid, y_valid, y_valid_pred, sens_feature_idx)

        valid_pred = valid.copy()
        valid_pred.labels = y_valid_pred
        fairness_metrics = ClassificationMetric(valid, valid_pred, unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups)

        balanced_accuracy = (fairness_metrics.true_positive_rate() + fairness_metrics.true_negative_rate()) / 2
        disparate_impact = fairness_metrics.disparate_impact()
        statistical_parity_difference = fairness_metrics.statistical_parity_difference()
        equal_opportunity_difference = fairness_metrics.equal_opportunity_difference()
        average_odds_difference = fairness_metrics.average_odds_difference()

        lambdas.append(lam)
        accuracies.append(accuracy_total)
        avg_odds_difs.append(average_odds_difference)

        if accuracy_total > accuracy_most_accurate:
            models_for_report['most accurate'] = lam
            accuracy_most_accurate = accuracy_total

        if average_odds_difference >= average_odds_dif_most_fair:
            models_for_report['most fair'] = lam
            average_odds_dif_most_fair = average_odds_difference

        if print_cross_val_conf_matrices_on:
            print(100 * "-")
            print("λ =", lam)
            print("Total confusion matrix")
            print(confusion_matrix_total)
            print("Privileged group confusion matrix")
            print(confusion_matrix_priv)
            print("Unprivileged group confusion matrix")
            print(confusion_matrix_unpriv)
            print("balanced_accuracy =", balanced_accuracy)
            print("disparate_impact =", disparate_impact)
            print("statistical_parity_difference =", statistical_parity_difference)
            print("equal_opportunity_difference =", equal_opportunity_difference)
            print("average_odds_difference =", average_odds_difference)
            print()
            print("total_accuracy", accuracy_total)

    plot(lambdas, "λ", accuracies, "Total Accuracy", avg_odds_difs, "Average Odds Difference", classifier_type, classifier_type)

    return models_for_report


def train_test_final_models(model_hyperparams: dict, classifier_type: str, data: tuple, sensitive_metadata: tuple):

    X_train, y_train, w_train, X_test, y_test, X_valid, y_valid, X_train_valid, y_train_valid, train, valid, test, train_valid = data
    sens_feature_idx, privileged_groups, unprivileged_groups = sensitive_metadata
    reweigh_on = not all(w_train == 1.)

    trained_models = []
    for lam in model_hyperparams.values():
        C = 1 / lam

        if classifier_type == "LogReg":
            model = LogisticRegression(solver='liblinear', random_state=15, C=C)
        elif classifier_type == "SVC":
            model = SVC(C=C)
        else:
            raise ValueError("Invalid classifier type!")

        if reweigh_on:
            model.fit(X_train, y_train, sample_weight=w_train)
        else:
            model.fit(X_train_valid, y_train_valid)

        trained_models.append(model)
        y_test_pred = model.predict(X_test)

        confusion_matrix_total = confusion_matrix(y_test, y_test_pred)
        accuracy_total = accuracy_score(y_test, y_test_pred)
        confusion_matrix_priv, confusion_matrix_unpriv = confusion_matrix_per_class(X_test, y_test, y_test_pred, sens_feature_idx)

        test_pred = test.copy()
        test_pred.labels = y_test_pred
        fairness_metrics = ClassificationMetric(test, test_pred, unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups)

        balanced_accuracy = (fairness_metrics.true_positive_rate() + fairness_metrics.true_negative_rate()) / 2
        disparate_impact = fairness_metrics.disparate_impact()
        statistical_parity_difference = fairness_metrics.statistical_parity_difference()
        equal_opportunity_difference = fairness_metrics.equal_opportunity_difference()
        average_odds_difference = fairness_metrics.average_odds_difference()

        print(100 * "-")
        print("λ =", lam)
        print("Total confusion matrix")
        print(confusion_matrix_total)
        print("Privileged group confusion matrix")
        print(confusion_matrix_priv)
        print("Unprivileged group confusion matrix")
        print(confusion_matrix_unpriv)
        print("balanced_accuracy =", balanced_accuracy)
        print("disparate_impact =", disparate_impact)
        print("statistical_parity_difference =", statistical_parity_difference)
        print("equal_opportunity_difference =", equal_opportunity_difference)
        print("average_odds_difference =", average_odds_difference)
        print()
        print("total_accuracy", accuracy_total)

    return trained_models


def train_test_final_models_RF(model_hyperparams: dict, classifier_type: str, data: tuple, sensitive_metadata: tuple):

    X_train, y_train, w_train, X_test, y_test, X_valid, y_valid, X_train_valid, y_train_valid, train, valid, test, train_valid = data
    sens_feature_idx, privileged_groups, unprivileged_groups = sensitive_metadata
    reweigh_on = not all(w_train == 1.)

    trained_models = []
    for max_depth in model_hyperparams.values():

        if classifier_type == "RF":
            model = RandomForestClassifier(n_estimators=500, max_depth=max_depth)
        else:
            raise ValueError("Invalid classifier type!")

        if reweigh_on:
            model.fit(X_train, y_train, sample_weight=w_train)
        else:
            model.fit(X_train_valid, y_train_valid)
        trained_models.append(model)
        y_test_pred = model.predict(X_test)

        confusion_matrix_total = confusion_matrix(y_test, y_test_pred)
        accuracy_total = accuracy_score(y_test, y_test_pred)
        confusion_matrix_priv, confusion_matrix_unpriv = confusion_matrix_per_class(X_test, y_test, y_test_pred, sens_feature_idx)

        test_pred = test.copy()
        test_pred.labels = y_test_pred
        fairness_metrics = ClassificationMetric(test, test_pred, unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups)

        balanced_accuracy = (fairness_metrics.true_positive_rate() + fairness_metrics.true_negative_rate()) / 2
        disparate_impact = fairness_metrics.disparate_impact()
        statistical_parity_difference = fairness_metrics.statistical_parity_difference()
        equal_opportunity_difference = fairness_metrics.equal_opportunity_difference()
        average_odds_difference = fairness_metrics.average_odds_difference()

        print(100 * "-")
        print("max depth =", max_depth)
        print("Total confusion matrix")
        print(confusion_matrix_total)
        print("Privileged group confusion matrix")
        print(confusion_matrix_priv)
        print("Unprivileged group confusion matrix")
        print(confusion_matrix_unpriv)
        print("balanced_accuracy =", balanced_accuracy)
        print("disparate_impact =", disparate_impact)
        print("statistical_parity_difference =", statistical_parity_difference)
        print("equal_opportunity_difference =", equal_opportunity_difference)
        print("average_odds_difference =", average_odds_difference)
        print()
        print("total_accuracy", accuracy_total)

    return trained_models


def max_depth_grid_search(max_depths: list, classifier_type: str, data: tuple, sensitive_metadata: tuple,
                          print_cross_val_conf_matrices_on: bool):

    X_train, y_train, w_train, X_test, y_test, X_valid, y_valid, X_train_valid, y_train_valid, train, valid, test, train_valid = data
    sens_feature_idx, privileged_groups, unprivileged_groups = sensitive_metadata

    models_for_report = dict()
    average_odds_dif_most_fair = -float('inf')
    accuracy_most_accurate = 0

    accuracies = []
    avg_odds_difs = []

    # Train many models for different hyperparameter value, print the accuracy and fairness results of each one of them,
    # (if print_on), and note the hyperparameters of the most accurate and most fair models
    for max_depth in max_depths:

        if classifier_type == "RF":
            model = RandomForestClassifier(n_estimators=500, max_depth=max_depth)
        else:
            raise ValueError("Invalid classifier type!")

        model.fit(X_train, y_train, sample_weight=w_train)
        y_valid_pred = model.predict(X_valid)

        confusion_matrix_total = confusion_matrix(y_valid, y_valid_pred)
        accuracy_total = accuracy_score(y_valid, y_valid_pred)
        confusion_matrix_priv, confusion_matrix_unpriv = confusion_matrix_per_class(X_valid, y_valid, y_valid_pred, sens_feature_idx)

        valid_pred = valid.copy()
        valid_pred.labels = y_valid_pred
        fairness_metrics = ClassificationMetric(valid, valid_pred, unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups)

        balanced_accuracy = (fairness_metrics.true_positive_rate() + fairness_metrics.true_negative_rate()) / 2
        disparate_impact = fairness_metrics.disparate_impact()
        statistical_parity_difference = fairness_metrics.statistical_parity_difference()
        equal_opportunity_difference = fairness_metrics.equal_opportunity_difference()
        average_odds_difference = fairness_metrics.average_odds_difference()

        accuracies.append(accuracy_total)
        avg_odds_difs.append(average_odds_difference)

        if accuracy_total > accuracy_most_accurate:
            models_for_report['most accurate'] = max_depth
            accuracy_most_accurate = accuracy_total

        if average_odds_difference >= average_odds_dif_most_fair:
            models_for_report['most fair'] = max_depth
            average_odds_dif_most_fair = average_odds_difference

        if print_cross_val_conf_matrices_on:
            print(100 * "-")
            print("max_depth =", max_depth)
            print("Total confusion matrix")
            print(confusion_matrix_total)
            print("Privileged group confusion matrix")
            print(confusion_matrix_priv)
            print("Unprivileged group confusion matrix")
            print(confusion_matrix_unpriv)
            print("balanced_accuracy =", balanced_accuracy)
            print("disparate_impact =", disparate_impact)
            print("statistical_parity_difference =", statistical_parity_difference)
            print("equal_opportunity_difference =", equal_opportunity_difference)
            print("average_odds_difference =", average_odds_difference)
            print()
            print("total_accuracy", accuracy_total)

    plot(max_depths, "Max Depth", accuracies, "Total Accuracy", avg_odds_difs, "Average Odds Difference", classifier_type, classifier_type)

    return models_for_report


def lambda_grid_search_cross_val(lambda_values: list, classifier_type: str, data: tuple, sensitive_metadata: tuple,
                                 print_cross_val_conf_matrices_on: bool):

    X_train, y_train, w_train, X_test, y_test, X_valid, y_valid, X_train_valid, y_train_valid, train, valid, test, train_valid = data
    sens_feature_idx, privileged_groups, unprivileged_groups = sensitive_metadata
    reweigh_on = not all(w_train==1.)

    # Prepare k-splits
    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=2)

    models_for_report = dict()
    average_odds_dif_most_fair = -float('inf')
    accuracy_most_accurate = 0

    lambdas = []

    accuracies_avgs = []
    confusion_matrices_total_sums = []
    confusion_matrices_priv_sums = []
    confusion_matrices_unpriv_sums = []
    balanced_accuracies_avgs = []
    disparate_impacts_avgs = []
    statistical_parity_differences_avgs = []
    equal_opportunity_differences_avgs = []
    average_odds_differences_avgs = []

    # Train many models for different hyperparameter value, print the accuracy and fairness results of each one of them,
    # (if print_on), and note the hyperparameters of the most accurate and most fair models
    for idx, lam in enumerate(lambda_values):
        print("Working on λ #{}".format(idx))

        train_valid_split_iterator = kf.split(X_train_valid, y_train_valid)

        C = 1 / lam

        if classifier_type == "LogReg":
            model = LogisticRegression(solver='liblinear', random_state=8, C=C)
        elif classifier_type == "SVC":
            model = SVC(C=C)
        else:
            raise ValueError("Invalid classifier type!")

        accuracies_perfold = []
        confusion_matrices_total_perfold = []
        confusion_matrices_priv_perfold = []
        confusion_matrices_unpriv_perfold = []
        balanced_accuracies_perfold = []
        disparate_impacts_perfold = []
        statistical_parity_differences_perfold = []
        equal_opportunity_differences_perfold = []
        average_odds_differences_perfold = []

        for fold in range(0, num_folds):
            train_valid_idcs = next(train_valid_split_iterator)
            train = train_valid.subset(train_valid_idcs[0])
            valid = train_valid.subset(train_valid_idcs[1])
            X_valid = X_train_valid[train_valid_idcs[1]]
            y_valid = y_train_valid[train_valid_idcs[1]]

            if reweigh_on:
                RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
                RW.fit(train)
                train = RW.transform(train)

            X_train = train.features
            y_train = train.labels.ravel()
            w_train = train.instance_weights.ravel()

            model.fit(X_train, y_train, sample_weight=w_train)
            y_valid_pred = model.predict(X_valid)

            confusion_matrix_total = confusion_matrix(y_valid, y_valid_pred)
            accuracy_total = accuracy_score(y_valid, y_valid_pred)
            confusion_matrix_priv, confusion_matrix_unpriv = confusion_matrix_per_class(X_valid, y_valid, y_valid_pred, sens_feature_idx)

            valid_pred = valid.copy()
            valid_pred.labels = y_valid_pred
            fairness_metrics = ClassificationMetric(valid, valid_pred, unprivileged_groups=unprivileged_groups,
                                                    privileged_groups=privileged_groups)

            balanced_accuracy = (fairness_metrics.true_positive_rate() + fairness_metrics.true_negative_rate()) / 2
            disparate_impact = fairness_metrics.disparate_impact()
            statistical_parity_difference = fairness_metrics.statistical_parity_difference()
            equal_opportunity_difference = fairness_metrics.equal_opportunity_difference()
            average_odds_difference = fairness_metrics.average_odds_difference()

            accuracies_perfold.append(accuracy_total)
            confusion_matrices_total_perfold.append(confusion_matrix_total)
            confusion_matrices_priv_perfold.append(confusion_matrix_priv)
            confusion_matrices_unpriv_perfold.append(confusion_matrix_unpriv)
            balanced_accuracies_perfold.append(balanced_accuracy)
            disparate_impacts_perfold.append(disparate_impact)
            statistical_parity_differences_perfold.append(statistical_parity_difference)
            equal_opportunity_differences_perfold.append(equal_opportunity_difference)
            average_odds_differences_perfold.append(average_odds_difference)

        accuracies_avg = sum(accuracies_perfold) / len(accuracies_perfold)
        confusion_matrices_total_sum = sum(confusion_matrices_total_perfold)
        confusion_matrices_priv_sum = sum(confusion_matrices_priv_perfold)
        confusion_matrices_unpriv_sum = sum(confusion_matrices_unpriv_perfold)
        balanced_accuracies_avg = sum(balanced_accuracies_perfold) / len(balanced_accuracies_perfold)
        disparate_impacts_avg = sum(disparate_impacts_perfold) / len(disparate_impacts_perfold)
        statistical_parity_differences_avg = sum(statistical_parity_differences_perfold) / len(statistical_parity_differences_perfold)
        equal_opportunity_differences_avg = sum(equal_opportunity_differences_perfold) / len(equal_opportunity_differences_perfold)
        average_odds_differences_avg = sum(average_odds_differences_perfold) / len(average_odds_differences_perfold)

        accuracies_avgs.append(accuracies_avg)
        confusion_matrices_total_sums.append(confusion_matrices_total_sum)
        confusion_matrices_priv_sums.append(confusion_matrices_priv_sum)
        confusion_matrices_unpriv_sums.append(confusion_matrices_unpriv_sum)
        balanced_accuracies_avgs.append(balanced_accuracies_avg)
        disparate_impacts_avgs.append(disparate_impacts_avg)
        statistical_parity_differences_avgs.append(statistical_parity_differences_avg)
        equal_opportunity_differences_avgs.append(equal_opportunity_differences_avg)
        average_odds_differences_avgs.append(average_odds_differences_avg)

        lambdas.append(lam)

        if accuracies_avg > accuracy_most_accurate:
            models_for_report['most accurate'] = lam
            accuracy_most_accurate = accuracies_avg

        if average_odds_differences_avg >= average_odds_dif_most_fair:
            models_for_report['most fair'] = lam
            average_odds_dif_most_fair = average_odds_differences_avg

        if print_cross_val_conf_matrices_on:
            print(100 * "-")
            print("λ =", lam)
            print("Total confusion matrix")
            print(confusion_matrices_total_sum)
            print("Privileged group confusion matrix")
            print(confusion_matrices_priv_sum)
            print("Unprivileged group confusion matrix")
            print(confusion_matrices_unpriv_sum)
            print("balanced_accuracy =", balanced_accuracies_avg)
            print("disparate_impact =", disparate_impacts_avg)
            print("statistical_parity_difference =", statistical_parity_differences_avg)
            print("equal_opportunity_difference =", equal_opportunity_differences_avg)
            print("average_odds_difference =", average_odds_differences_avg)
            print()
            print("total_accuracy", accuracies_avg)

    plot(lambdas, "λ", accuracies_avgs, "Total Accuracy", average_odds_differences_avgs, "Average Odds Difference", classifier_type, classifier_type)

    return models_for_report


def max_depth_grid_search_cross_val(max_depths: list, classifier_type: str, data: tuple, sensitive_metadata: tuple,
                                    print_cross_val_conf_matrices_on: bool):

    X_train, y_train, w_train, X_test, y_test, X_valid, y_valid, X_train_valid, y_train_valid, train, valid, test, train_valid = data
    sens_feature_idx, privileged_groups, unprivileged_groups = sensitive_metadata
    reweigh_on = not all(w_train == 1.)

    # Prepare k-splits
    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=2)

    models_for_report = dict()
    average_odds_dif_most_fair = -float('inf')
    accuracy_most_accurate = 0

    accuracies_avgs = []
    confusion_matrices_total_sums = []
    confusion_matrices_priv_sums = []
    confusion_matrices_unpriv_sums = []
    balanced_accuracies_avgs = []
    disparate_impacts_avgs = []
    statistical_parity_differences_avgs = []
    equal_opportunity_differences_avgs = []
    average_odds_differences_avgs = []

    # Train many models for different hyperparameter value, print the accuracy and fairness results of each one of them,
    # (if print_on), and note the hyperparameters of the most accurate and most fair models
    for idx, max_depth in enumerate(max_depths):

        train_valid_split_iterator = kf.split(X_train_valid, y_train_valid)

        if classifier_type == "RF":
            model = RandomForestClassifier(n_estimators=500, max_depth=max_depth)
        else:
            raise ValueError("Invalid classifier type!")

        accuracies_perfold = []
        confusion_matrices_total_perfold = []
        confusion_matrices_priv_perfold = []
        confusion_matrices_unpriv_perfold = []
        balanced_accuracies_perfold = []
        disparate_impacts_perfold = []
        statistical_parity_differences_perfold = []
        equal_opportunity_differences_perfold = []
        average_odds_differences_perfold = []

        for fold in range(0, num_folds):
            train_valid_idcs = next(train_valid_split_iterator)
            train = train_valid.subset(train_valid_idcs[0])
            valid = train_valid.subset(train_valid_idcs[1])
            X_valid = X_train_valid[train_valid_idcs[1]]
            y_valid = y_train_valid[train_valid_idcs[1]]

            if reweigh_on:
                RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
                RW.fit(train)
                train = RW.transform(train)

            X_train = train.features
            y_train = train.labels.ravel()
            w_train = train.instance_weights.ravel()

            model.fit(X_train, y_train, sample_weight=w_train)
            y_valid_pred = model.predict(X_valid)

            confusion_matrix_total = confusion_matrix(y_valid, y_valid_pred)
            accuracy_total = accuracy_score(y_valid, y_valid_pred)
            confusion_matrix_priv, confusion_matrix_unpriv = confusion_matrix_per_class(X_valid, y_valid, y_valid_pred,
                                                                                        sens_feature_idx)

            valid_pred = valid.copy()
            valid_pred.labels = y_valid_pred
            fairness_metrics = ClassificationMetric(valid, valid_pred, unprivileged_groups=unprivileged_groups,
                                                    privileged_groups=privileged_groups)

            balanced_accuracy = (fairness_metrics.true_positive_rate() + fairness_metrics.true_negative_rate()) / 2
            disparate_impact = fairness_metrics.disparate_impact()
            statistical_parity_difference = fairness_metrics.statistical_parity_difference()
            equal_opportunity_difference = fairness_metrics.equal_opportunity_difference()
            average_odds_difference = fairness_metrics.average_odds_difference()

            accuracies_perfold.append(accuracy_total)
            confusion_matrices_total_perfold.append(confusion_matrix_total)
            confusion_matrices_priv_perfold.append(confusion_matrix_priv)
            confusion_matrices_unpriv_perfold.append(confusion_matrix_unpriv)
            balanced_accuracies_perfold.append(balanced_accuracy)
            disparate_impacts_perfold.append(disparate_impact)
            statistical_parity_differences_perfold.append(statistical_parity_difference)
            equal_opportunity_differences_perfold.append(equal_opportunity_difference)
            average_odds_differences_perfold.append(average_odds_difference)

        accuracies_avg = sum(accuracies_perfold) / len(accuracies_perfold)
        confusion_matrices_total_sum = sum(confusion_matrices_total_perfold)
        confusion_matrices_priv_sum = sum(confusion_matrices_priv_perfold)
        confusion_matrices_unpriv_sum = sum(confusion_matrices_unpriv_perfold)
        balanced_accuracies_avg = sum(balanced_accuracies_perfold) / len(balanced_accuracies_perfold)
        disparate_impacts_avg = sum(disparate_impacts_perfold) / len(disparate_impacts_perfold)
        statistical_parity_differences_avg = sum(statistical_parity_differences_perfold) / len(
            statistical_parity_differences_perfold)
        equal_opportunity_differences_avg = sum(equal_opportunity_differences_perfold) / len(
            equal_opportunity_differences_perfold)
        average_odds_differences_avg = sum(average_odds_differences_perfold) / len(average_odds_differences_perfold)

        accuracies_avgs.append(accuracies_avg)
        confusion_matrices_total_sums.append(confusion_matrices_total_sum)
        confusion_matrices_priv_sums.append(confusion_matrices_priv_sum)
        confusion_matrices_unpriv_sums.append(confusion_matrices_unpriv_sum)
        balanced_accuracies_avgs.append(balanced_accuracies_avg)
        disparate_impacts_avgs.append(disparate_impacts_avg)
        statistical_parity_differences_avgs.append(statistical_parity_differences_avg)
        equal_opportunity_differences_avgs.append(equal_opportunity_differences_avg)
        average_odds_differences_avgs.append(average_odds_differences_avg)

        if accuracies_avg > accuracy_most_accurate:
            models_for_report['most accurate'] = max_depth
            accuracy_most_accurate = accuracies_avg

        if average_odds_differences_avg >= average_odds_dif_most_fair:
            models_for_report['most fair'] = max_depth
            average_odds_dif_most_fair = average_odds_differences_avg

        if print_cross_val_conf_matrices_on:
            print(100 * "-")
            print("max_depth =", max_depth)
            print("Total confusion matrix")
            print(confusion_matrices_total_sum)
            print("Privileged group confusion matrix")
            print(confusion_matrices_priv_sum)
            print("Unprivileged group confusion matrix")
            print(confusion_matrices_unpriv_sum)
            print("balanced_accuracy =", balanced_accuracies_avg)
            print("disparate_impact =", disparate_impacts_avg)
            print("statistical_parity_difference =", statistical_parity_differences_avg)
            print("equal_opportunity_difference =", equal_opportunity_differences_avg)
            print("average_odds_difference =", average_odds_differences_avg)
            print()
            print("total_accuracy", accuracies_avg)

    plot(max_depths, "Max Depth", accuracies_avgs, "Total Accuracy", average_odds_differences_avgs, "Average Odds Difference", classifier_type, classifier_type)

    return models_for_report