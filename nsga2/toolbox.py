import os
import collections
import re
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from crepes import WrapClassifier

n_cpu = os.cpu_count()

def decode(classifier, var_range, **features):
    """
    Decoding hyperaparameters.
    """
    if classifier == "tree":
        features['criterion'] = round(features['criterion'], 0)
    
        if features['max_depth'] is not None:
            features['max_depth'] = int(round(features['max_depth']))
        #else:
        #    features['max_depth'] = var_range[1][1]
    
        features['min_samples_split'] = int(round(features['min_samples_split']))
    
        #features['min_samples_leaf'] = int(round(features['min_samples_leaf']))
    
        if features['max_leaf_nodes'] is not None:
            features['max_leaf_nodes'] = int(round(features['max_leaf_nodes']))
        #else:
        #    features['max_leaf_nodes'] = var_range[3][1]
    
        hyperparameters = ['criterion', 'max_depth', 'min_samples_split', 'max_leaf_nodes']
    
    elif classifier == "logit":
        features["max_iter"] = int(round(features["max_iter"], 0))
        

        hyperparameters = ["max_iter", "tol", "C", "l1_ratio"] 

    elif classifier == "rf":

        features["n_estimators"] = int(round(features["n_estimators"], 0))
        features['criterion'] = round(features['criterion'], 0)
    
        if features['max_depth'] is not None:
            features['max_depth'] = int(round(features['max_depth']))
    
        features['min_samples_split'] = int(round(features['min_samples_split']))
    
        #features['min_samples_leaf'] = int(round(features['min_samples_leaf']))
    
        if features['max_leaf_nodes'] is not None:
            features['max_leaf_nodes'] = int(round(features['max_leaf_nodes']))

    
        hyperparameters = ['n_estimators', 'max_features', 'max_samples', 'criterion', 'max_depth', 'min_samples_split', 'max_leaf_nodes']
    
    list_of_hyperparameters = [(hyperparameter, features[hyperparameter]) for hyperparameter in hyperparameters]
    features = collections.OrderedDict(list_of_hyperparameters)
    return features

def train_cal_val_test_split(X, sens, y, sizes=[.4, .1, .3, .2], stratify=True):
    """
    Split dataframe into training, calibration, validation and test sets.
    """
    if stratify:
        X_dev, X_test, sens_dev, sens_test, y_dev, y_test = train_test_split(X, sens, y, test_size=int(len(y)*sizes[3]), stratify=pd.concat([sens, y], axis=1))
        X_train, X_val, sens_train, sens_val, y_train, y_val = train_test_split(X_dev, sens_dev, y_dev, test_size=int(len(y)*sizes[2]), stratify=pd.concat([sens_dev, y_dev], axis=1))
        X_learn, X_cal, sens_learn, sens_cal, y_learn, y_cal = train_test_split(X_train, sens_train, y_train, test_size=int(len(y)*sizes[1]), stratify=pd.concat([sens_train, y_train], axis=1))
        return X_learn, X_cal, X_val, X_test, sens_learn, sens_cal, sens_val, sens_test, y_learn, y_cal, y_val, y_test
    else:
        X_dev, X_test, sens_dev, sens_test, y_dev, y_test = train_test_split(X, sens, y, test_size=int(len(y)*sizes[3]))
        X_train, X_val, sens_train, sens_val, y_train, y_val = train_test_split(X_dev, sens_dev, y_dev, test_size=int(len(y)*sizes[2]))
        X_learn, X_cal, sens_learn, sens_cal, y_learn, y_cal = train_test_split(X_train, y_train, test_size=int(len(y)*sizes[1]))
        return X_learn, X_cal, X_val, X_test, sens_learn, sens_cal, sens_val, sens_test, y_learn, y_cal, y_val, y_test

def learn_logit(X_learn, y_learn, **hyperparameters):
    """
    Learn a logistic regression classifier
    """
    logit = LogisticRegression(
        max_iter = hyperparameters["max_iter"],
        tol = hyperparameters["tol"],
        #penalty="elasticnet",
        C = hyperparameters['C'],
        l1_ratio=hyperparameters["l1_ratio"]
    )

    logit.fit(X_learn.values, y_learn.values)
    return logit

def learn_default_logit(X_train, y_train):
    """
    Learn a logistic regression classifier with default hyperparameters values.
    """
    logit = LogisticRegression()

    logit.fit(X_train.values, y_train.values)
    return logit

def learn_decision_tree(X_learn, y_learn, **hyperparameters):
    """
    Learn a decision tree classifier
    """
    if hyperparameters["criterion"] < 0.5:
        tree = DecisionTreeClassifier(
            criterion="gini",
            max_depth = hyperparameters['max_depth'],
            min_samples_split = hyperparameters['min_samples_split'],
            max_leaf_nodes = hyperparameters['max_leaf_nodes'],
        )

    else:
        tree = DecisionTreeClassifier(
            criterion="entropy",
            max_depth = hyperparameters['max_depth'],
            min_samples_split = hyperparameters['min_samples_split'],
            max_leaf_nodes = hyperparameters['max_leaf_nodes'],
        )

    tree.fit(X_learn.values, y_learn.values)
    return tree

def learn_default_decision_tree(X_train, y_train):
    """
    Learn a decision tree classifier with default hyperparameters values, except max_depth=10.
    """
    tree = DecisionTreeClassifier(max_depth=10)

    tree.fit(X_train.values, y_train.values)
    return tree

def learn_random_forest(X_learn, y_learn, **hyperparameters):
    """
    Learn a Random Forest classifier
    """
    
    if hyperparameters["criterion"] < 0.5:
        rf = RandomForestClassifier(
            n_estimators=hyperparameters['n_estimators'],
            max_features=hyperparameters['max_features'],
            max_samples=hyperparameters['max_samples'],
            criterion="gini",
            max_depth = hyperparameters['max_depth'],
            min_samples_split = hyperparameters['min_samples_split'],
            max_leaf_nodes = hyperparameters['max_leaf_nodes'],
            oob_score=False,
            n_jobs=-1
        )

    else:
        rf = RandomForestClassifier(
            criterion="entropy",
            n_estimators=hyperparameters['n_estimators'],
            max_features=hyperparameters['max_features'],
            max_samples=hyperparameters['max_samples'],
            max_depth = hyperparameters['max_depth'],
            min_samples_split = hyperparameters['min_samples_split'],
            max_leaf_nodes = hyperparameters['max_leaf_nodes'],
            oob_score=False
        )

    rf.fit(X_learn.values, y_learn.values)
    return rf

def learn_default_random_forest(X_train, y_train):
    """
    Learn a Random Forest classifier with default hyperparameters values, except max_depth=10.
    """
    tree = RandomForestClassifier(max_depth=10, oob_score=False)

    tree.fit(X_train.values, y_train.values)
    return tree

def calibrate_classifier(classifier, X_cal, y_cal, taxonomy=None):
    """
    Calibrate a classifier using a hold-out calibration set. A Mondrian taxonomy can be provided.
    """
    conformal_predictor = WrapClassifier(classifier)

    if taxonomy is not None:
        conformal_predictor.calibrate(X_cal.values, y_cal.values, bins=taxonomy.values)
    else:
        conformal_predictor.calibrate(X_cal.values, y_cal.values)
    return conformal_predictor

def calibrate_random_forest(classifier, X_train, y_train, taxonomy=None):
    """
    Calibrate a Random Forest classifier using out-of-bag predictions. A Mondrian taxonomy can be provided.
    """
    conformal_predictor = WrapClassifier(classifier)

    if taxonomy is not None:
        conformal_predictor.calibrate(X_train.values, y_train.values, bins=taxonomy.values, oob=False)
    else:
        conformal_predictor.calibrate(X_train.values, y_train.values, oob=False)
    
    return conformal_predictor


def marginal_coverage(icp, X, y, confidence, taxonomy=None):
    """
    Compute marginal coverage metric for a given conformal predictor.
    """
    if taxonomy is not None:
        return 1 - icp.evaluate(X.values, y.values, confidence=confidence, bins=taxonomy.values)["error"]
    else:
        return 1 - icp.evaluate(X.values, y.values, confidence=confidence)["error"]

def average_set_size(icp, X, y, confidence, taxonomy=None):
    """
    Compute average set size for a given conformal predictor.
    """
    if taxonomy is not None:
        return icp.evaluate(X.values, y.values, confidence=confidence, bins=taxonomy.values)["avg_c"]
    else:
        return icp.evaluate(X.values, y.values, confidence=confidence)["avg_c"]

def equalized_coverage_gap(icp, X_priv, X_unpriv, y_priv, y_unpriv, confidence):
    """
    Compute coverage gap between priviledged and unpriviledged groups.
    """
    priv_cov = 1 - icp.evaluate(X_priv.values, y_priv.values, confidence=confidence)["error"]
    unpriv_cov = 1 - icp.evaluate(X_unpriv.values, y_unpriv.values, confidence=confidence)["error"]
    return np.abs(priv_cov - unpriv_cov)

def group_specific_coverage(icp, X, y, sens, confidence, taxonomy=None):
    """
    Compute group-conditional coverages.
    """
    group_coverage_keys = ["cov_" + str(s) for s in sens.unique()]
    group_coverage = dict.fromkeys(group_coverage_keys)

    for cov in group_coverage.keys():
        group = cov[-1]
        X_group = X[sens == int(group)]
        y_group = y[sens == int(group)]
        if taxonomy is not None:
            group_coverage[cov] = 1 - icp.evaluate(X_group.values, y_group.values, confidence=confidence, bins=taxonomy.values)["error"]
        else:
            group_coverage[cov] = 1 - icp.evaluate(X_group.values, y_group.values, confidence=confidence)["error"]


    return group_coverage

def avg_priv_dif(group_cov_dict, group):
    priv_key = f"cov_{group}"
    unpriv_keys = [key for key in group_cov_dict.keys() if key != priv_key]

    differences = [abs(group_cov_dict[priv_key] - group_cov_dict[other_group]) for other_group in unpriv_keys]
    
    average_dif = sum(differences) / len(differences)

    return average_dif

def max_priv_dif(group_cov_dict, group):
    priv_key = f"cov_{group}"
    unpriv_keys = [key for key in group_cov_dict.keys() if key != priv_key]

    differences = [abs(group_cov_dict[priv_key] - group_cov_dict[other_group]) for other_group in unpriv_keys]
    
    max_dif = max(differences)

    return max_dif

def group_specific_efficiency(icp, X, y, sens, confidence, taxonomy=None):
    """
    Compute group-conditional efficiency.
    """
    group_eff_keys = ["eff_" + str(s) for s in sens.unique()]
    group_eff = dict.fromkeys(group_eff_keys)

    for eff in group_eff.keys():
        group = eff[-1]
        X_group = X[sens == int(group)]
        y_group = y[sens == int(group)]
        if taxonomy is not None:
            group_eff[eff] = icp.evaluate(X_group.values, y_group.values, confidence=confidence, bins=taxonomy.values)["avg_c"]
        else:
            group_eff[eff] = icp.evaluate(X_group.values, y_group.values, confidence=confidence)["avg_c"]
    
    return group_eff

def results_from_experiment(dataset, fair_obj, model, sens, calsize):

    files = os.listdir("./results/" + dataset + "/" + fair_obj + "/" + model + "/individuals/")
    li = []


    for file in files:
        if f"individuals_pareto_{sens}_calsize_{calsize}" in file:
            df = pd.read_csv("./results/" + dataset + "/" + fair_obj + "/" + model + "/individuals/" + file, index_col=None, header=0)
            df = df[~df.duplicated(subset=['creation_mode', 'set_size_val', "unfairness_val"], keep="last")]
            df["execution"] = re.search(r'_seed_(\d+)_', file).group(1)
            li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)

    pareto_solutions = frame[frame["creation_mode"] == "mo_opt"]
    
    icp = frame[frame["creation_mode"] == "icp"]
    micp = frame[frame["creation_mode"] == "micp"]

    return pareto_solutions, icp, micp

def pareto_from_experiment(dataset, fair_obj, model, sens, calsize):

    files = os.listdir("./results/" + dataset + "/" + fair_obj + "/" + model + "/individuals/")

    li = []


    for file in files:
        if f"individuals_pareto_{sens}_calsize_{calsize}" in file:
            df = pd.read_csv("./results/" + dataset + "/" + fair_obj + "/" + model + "/individuals/" + file, index_col=None, header=0)
            df = df[~df.duplicated(subset=['creation_mode', 'set_size_val', "unfairness_val"], keep="last")]
            df["execution"] = re.search(r'_seed_(\d+)_', file).group(1)
            li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)

    pareto_solutions = frame[frame["creation_mode"] == "mo_opt"]

    return pareto_solutions

def compute_average_pareto(pareto_solutions, set="val", quartil=True):
    avg_solutions = round(np.mean(pareto_solutions["execution"].value_counts()))
    if quartil:
        percentiles = [0, 0.25, 0.5, 0.75, 1]
    else:
        percentiles = np.linspace(0, 1, avg_solutions)


    if set == "val":
        avg_pareto = pd.DataFrame(columns=["set_size_val", "unfairness_val"])
        for perc in percentiles:        
            avg_eff = np.mean(pareto_solutions.groupby('execution')["set_size_val"].quantile(perc))
            avg_unf = np.mean(pareto_solutions.groupby('execution')["unfairness_val"].quantile(1-perc))
            perc_info = {"set_size_val": [avg_eff], "unfairness_val": [avg_unf]}
            perc_aux = pd.DataFrame(perc_info)
            avg_pareto = pd.concat([avg_pareto, perc_aux])
    else:
        avg_pareto = pd.DataFrame(columns=["set_size_test", "unfairness_test"])
        for perc in percentiles:  
            avg_eff = np.mean(pareto_solutions.groupby('execution')["set_size_test"].quantile(perc))
            avg_unf = np.mean(pareto_solutions.groupby('execution')["unfairness_test"].quantile(1-perc))
            perc_info = {"set_size_test": [avg_eff], "unfairness_test": [avg_unf]}
            perc_aux = pd.DataFrame(perc_info)
            avg_pareto = pd.concat([avg_pareto, perc_aux])

    return avg_pareto

def results_conformal(icp, micp):
    icp_eff_val = np.mean(icp["set_size_val"])
    icp_unf_val = np.mean(icp["unfairness_val"])
    icp_eff_test = np.mean(icp["set_size_test"])
    icp_unf_test = np.mean(icp["unfairness_test"])
    micp_eff_val = np.mean(micp["set_size_val"])
    micp_unf_val = np.mean(micp["unfairness_val"])
    micp_eff_test = np.mean(micp["set_size_test"])
    micp_unf_test = np.mean(micp["unfairness_test"])

    icp_info =  {"set_size_val": [icp_eff_val], "unfairness_val": [icp_unf_val], "set_size_test": [icp_eff_test], "unfairness_test": [icp_unf_test]}
    micp_info =  {"set_size_val": [micp_eff_val], "unfairness_val": [micp_unf_val], "set_size_test": [micp_eff_test], "unfairness_test": [micp_unf_test]}

    conformal_results = pd.concat([pd.DataFrame(icp_info), pd.DataFrame(micp_info)])
    return conformal_results