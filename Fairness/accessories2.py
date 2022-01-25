from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing.reweighing import Reweighing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
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


def load_split_normalise_reweight_dataset(dataset: str, sensitive_feature: str,
                                          train_valid_test_proportions: list, print_dataset_sizes_on: bool,
                                          reweighing_on=True) -> tuple:

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
    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)
    RW.fit(train)
    train_transf = RW.transform(train)

    # Normalise train, test and valid datasets
    scaler = StandardScaler()

    if reweighing_on:
        scaler = scaler.fit(train_transf.features)
    else:
        scaler = scaler.fit(train.features)

    X_train = scaler.transform(train.features)
    y_train = train.labels.ravel()
    w_train = train.instance_weights.ravel()
    X_train_transf = scaler.transform(train_transf.features)
    y_train_transf = train_transf.labels.ravel()
    w_train_transf = train_transf.instance_weights.ravel()
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
        print("X_train_transf")
        print("type:", type(X_train_transf))
        print("shape:", X_train_transf.shape)
        print(100 * "-")
        print("y_train_transf")
        print("type:", type(y_train_transf))
        print("shape:", y_train_transf.shape)
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
        print("Weights of training samples before reweighing:", train.instance_weights)
        print("Weights of training samples after reweighing:", train_transf.instance_weights)
        print(100 * "-")
        print("Privileged Group Label =", test.favorable_label, "(Unprivileged Group Label = {})".format(test.unfavorable_label))
        print("Protected Attributes =", test.protected_attribute_names)
        print("Features =", test.feature_names)
        print(100 * "-")

    if reweighing_on:
        X_train_out = X_train_transf
        y_train_out = y_train_transf
        w_train_out = w_train_transf
    else:
        X_train_out = X_train
        y_train_out = y_train
        w_train_out = w_train

    return (X_train_out, y_train_out, w_train_out, X_test, y_test, X_valid, y_valid, X_train_valid, y_train_valid,
            train, train_transf, valid, test, sens_feature_idx, privileged_groups, unprivileged_groups)


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

    # Usually for this task the training data have been reweighed
    X_train, y_train, w_train, X_test, y_test, X_valid, y_valid, X_train_valid, y_train_valid, train, valid, test = data
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

    X_train, y_train, X_test, y_test, X_valid, y_valid, X_train_valid, y_train_valid, train, valid, test = data
    sens_feature_idx, privileged_groups, unprivileged_groups = sensitive_metadata

    trained_models = []
    for lam in model_hyperparams.values():
        C = 1 / lam

        if classifier_type == "LogReg":
            model = LogisticRegression(solver='liblinear', random_state=1, C=C)
        elif classifier_type == "SVC":
            model = SVC(C=C)
        else:
            raise ValueError("Invalid classifier type!")

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


def max_depth_grid_search(max_depths: list, classifier_type: str, data: tuple, sensitive_metadata: tuple,
                          print_cross_val_conf_matrices_on: bool):

    X_train, y_train, w_train, X_test, y_test, X_valid, y_valid, X_train_valid, y_train_valid, train, valid, test = data
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

