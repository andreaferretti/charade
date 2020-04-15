import os
import json
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split

import common.bert.logger as logger
import common.bert.constants as constants

import warnings
warnings.filterwarnings('ignore') # to ignore warnings on sklearn


log = logger.get_logger(logger.INFO)


def load_datasets(configs, path_to_config_folder, valid_size=0.2, random_state=0):
    """
    Loads data from file accordingly to configuration
    specified in config file.

    Parameters
    ----------
    configs : dict
        Configuration dictionary
    path_to_config_folder : str
        Path to configuration file
    valid_size : float,  default 0.2
        Dimension of validation dataset: if no validation dataset
        is provided within the config file, it is obtained from
        training dataset with random splitting
    random_state : int, default 0
        Random state

    Returns
    -------
    datasets : dict
        Dictionary containing datasets

    """

    get_path = lambda phase: os.path.join(path_to_config_folder, configs.get(constants.DATA_FOLDER), configs.get(phase))

    # loading data
    training_data = pd.read_csv(get_path(constants.TRAIN))
    if constants.VALID in configs:
        valid_data = pd.read_csv(get_path(constants.VALID))
    else:
        # if no validation path is specified, the training is splitted among actual training and validation datasets
        training_data, valid_data = train_test_split(training_data, random_state=random_state, test_size=valid_size)

    datasets = {constants.TRAIN: training_data, constants.VALID: valid_data}
    if constants.TEST in configs:  # test_data con be missing
        datasets[constants.TEST] = pd.read_csv(get_path(constants.TEST))

    return datasets


def load_configs(path_to_config_folder):
    """
    Loads configuration dictionary from file

    Parameters
    ----------
    path_to_config_folder : str
        Path to config file

    Returns
    -------
    configs : dict
        Configuration dictionary

    """

    with open(os.path.join(path_to_config_folder, constants.CONFIG_FILE), 'r') as f:
        configs = json.load(f)

    return configs


def load_labels(path_to_labels_folder):
    """
    Loads labels from file if available in
    {path_to_labels_folder}/{constants.LABELS_FILE}.

    Parameters
    ----------
    path_to_labels_folder : str
        Path to labels file

    Returns
    -------
    id2label : dict or None
        Labels dictionary or None

    """

    id2label = None
    if os.path.isdir(path_to_labels_folder) and constants.LABELS_FILE in os.listdir(path_to_labels_folder):
        with open(os.path.join(path_to_labels_folder, constants.LABELS_FILE), 'r') as f:
            id2label = json.load(f)

    return id2label


def init_learning_curves(metrics, phases=constants.PHASES):
    """
    Initializes learning curves as a dictionary mapping each
    phase in phased to a dictionary with keys constants.Metrics.loss
    and all values in metrics.

    Parameters
    ----------
    metrics : list, default [constants.Metrics.accuracy]
        List of metrics to monitor during training
    phases : list
        List of phases

    Returns
    -------
    learning_curves : dict
        Learning curves

    """

    learning_curves = {}
    for phase in phases:
        learning_curves[phase] = {}
        for m in [constants.METRICS.LOSS] + metrics:
            learning_curves[phase][m] = []

    return learning_curves


def update_learning_curves(learning_curves, results):
    """
    Updates learning curves with given results.

    Parameters
    ----------
    learning_curves : dict
        Learning curves
    results : dict
        Dictionary containing results of current epoch

    """

    for phase in results:
        for metric, value in results[phase].items():
            learning_curves[phase][metric].append(value)


def evaluate(y, y_pred, metric, average='binary'):
    """
    Evaluates current model on given labels and predictions
    accordingly to given metric.

    Parameters
    ----------
    y : pandas.DataFrame
        Labels
    y_pred : pandas.DataFrame
        Predictions
    metric : constants.METRICS
        Metric to monitor during training
    average : str, default 'binary'
        Parameter for computing metrics in case of multiclass/multilabel targets.

    Returns
    -------
    metric_result : float
        Metric result of given metric

    """

    metric_result = 0
    if metric == constants.METRICS.ACCURACY:
        metric_result = sklearn.metrics.accuracy_score(y, y_pred)
    elif metric == constants.METRICS.PRECISION:
        metric_result = sklearn.metrics.precision_score(y, y_pred, average = average)
    elif metric == constants.METRICS.RECALL:
        metric_result = sklearn.metrics.recall_score(y, y_pred, average = average)
    elif metric == constants.METRICS.F1_SCORE:
        metric_result = sklearn.metrics.f1_score(y, y_pred, average = average)
    else:
        log.warning("Metric {} not supported".format(metric))

    return metric_result
