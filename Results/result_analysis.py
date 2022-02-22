
import numpy as np
import pandas as pd
import pickle
import sys
import os
import sklearn.metrics as met
import matplotlib.pyplot as plt


def read_results(cfg):
    """
    Function to read the results returned from the eval_net.py function
    :param cfg: Config file
    :return: the results as dictionary
    """

    pcl = pickle.load(open(os.path.join(cfg.OUTPUT_DIR, 'result.pkl'),'rb'))
    return pcl

def setup_matrices(result_dict):
    """

    :param result_dict: (dict) containing results
    :return: np.ndarrays of predictions and labels
    """

    preds = np.zeros((len(result_dict, 2)))
    labels = np.zeros((len(result_dict, 1)))

    for idx, (key, val) in enumerate(result_dict.items()):
        assert len(val['predictions']) == len(val['labels'])

        if len(val['predictions']) == 1:
            preds[idx] = val['predictions']
            labels[idx] = val['labels']
        else:
            p = 0
            if np.allclose(*val['labels']) is False:
                raise ValueError("Something is wrong here, labels for all crops are not the same")
            labels[idx] = val['labels'][0]
            for i in val['predictions']:
                p += i
            preds[idx] = p

    return preds, labels

def get_simple_metrics(preds, labels):
    """

    :param preds: (np.ndarray) containing the predictions
    :param labels: (np.ndarray) containing the labels
    :return: (dict) containing relevant metrics
    """

    if preds.shape[-1] > 1:
        npreds = np.argmax(preds,axis = 1)
    else:
        npreds = preds.copy()

    if labels.shape[-1] > 1:
        nlabs = np.argmax(labels, axis =1)
    else:
        nlabs = labels.copy()


    metrics = {}

    metrics['weighted_accuracy'] = met.balanced_accuracy_score((nlabs, npreds))
    metrics['accuracy'] = met.accuracy_score((nlabs, npreds))
    metrics['confusion_matrix'] = conf_mat = met.confusion_matrix((nlabs, npreds))
    metrics['per_class_sensitivity'] = conf_mat.diagonal()/np.sum(conf_mat, axis = 1)

    return metrics


def calculate_roc_curve(preds, labels):
    """

    :param preds: (np.ndarray) containing the predictions
    :param labels: (np.ndarray) containing the labels
    :return: ((dict),)  first dict with fpr, second tpr, third roc_auc
    """

    if preds.shape[-1] <= 1:
        raise ValueError(f"Predictions should have shape[-1] > 1 instead of {preds.shape[-1]}")

    if labels.shape[-1] <= 1
        raise ValueError(f"labels should have shape[-1] > 1 instead of {labels.shape[-1]}")

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(preds.shape[-1]):
        fpr[i], tpr[i], _ = met.roc_curve(labels[:, i], preds[:, i])
        roc_auc[i] = met.auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = met.roc_curve(labels.ravel(), preds.ravel())
    roc_auc["micro"] = met.auc(fpr["micro"], tpr["micro"])

    return fpr, tpr, roc_auc


def plot_roc_curve(fpr, tpr,roc_auc, plot = True):
    """

    :param fpr: (dict) containing the false-positive rate for each class (keys)
    :param tpr: (dict) containing the true-positive rate for each class (keys)
    :param roc_auc: (dict) containing the areas under the curves
    :param plot: (bool) if True figure is plotted directly
    :return: None but creates the plt object
    """

    classes = ['Doner' , 'Fetal']
    lw = 2
    plt.figure()
    for idx, (key, val) in enumerate(fpr.items()):
        if key != "micro":
            plt.plot(fpr[key],
                         tpr[key],
                         lw=lw,
                         label = f"{classes[idx]} ROC Curve: AUC = {roc_auc[key]:.3f}"
                         )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    if plot:
        plt.show()







