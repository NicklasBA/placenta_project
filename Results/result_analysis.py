
import numpy as np
import pandas as pd
import pickle
import sys
import os
import sklearn.metrics as met
import matplotlib.pyplot as plt
import cv2

def read_results(cfg,  path = None):
    """
    Function to read the results returned from the eval_net.py function
    :param cfg: Config file
    :return: the results as dictionary
    """

    if path is None:
        pcl = pickle.load(open(os.path.join(cfg.OUTPUT_DIR, 'result.pkl'),'rb'))
    else:
        if os.path.isfile(path):
            pcl = pickle.load(open(path, 'rb'))
        elif os.path.isdir(path):
            if os.path.isfile(os.path.join(path, 'results.pkl')):
                pcl = pickle.load(open(os.path.join(path, 'results.pkl'),'rb'))
            else:
                raise ValueError("If path is directory, there should be a file named 'results.pkl' at location")
        else:
            raise AttributeError("Path is neither file or directory with appropriate file")

    return pcl

def setup_matrices(result_dict):
    """

    :param result_dict: (dict) containing results
    :return: np.ndarrays of predictions and labels
    """

    preds = np.zeros((len(result_dict), 2))
    labels = np.zeros((len(result_dict), 1))

    for idx, (key, val) in enumerate(result_dict.items()):
        assert len(val['predictions']) == len(val['labels'])

        if len(val['predictions']) == 1:
            if isinstance(val['predictions'][0], np.ndarray) is False:
                preds[idx] = val['predictions'][0].numpy()
            else:
                preds[idx] = val['predictions'][0]

            labels[idx] = val['labels'][0]
        else:
            raise ValueError(f"Length of labels should not {len(val['labels'])} be larger than 1 per video")

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

    metrics['weighted_accuracy'] = met.balanced_accuracy_score(nlabs, npreds)
    metrics['accuracy'] = met.accuracy_score(nlabs, npreds)
    metrics['confusion_matrix'] = conf_mat = met.confusion_matrix(nlabs, npreds)
    metrics['per_class_sensitivity'] = conf_mat.diagonal()/np.sum(conf_mat, axis = 1)

    return metrics

def setup_matrix_for_single(pcl):

    preds = np.array([val[0].numpy() if isinstance(val[0], np.ndarray) is False else val[0] for val in pcl['predictions']])
    labels = np.array([val[0] for val in pcl['labels']])[:, None]

    return preds, labels


def calculate_roc_curve(preds, labels):
    """

    :param preds: (np.ndarray) containing the predictions
    :param labels: (np.ndarray) containing the labels
    :return: ((dict),)  first dict with fpr, second tpr, third roc_auc
    """

    if preds.shape[-1] <= 1:
        raise ValueError(f"Predictions should have shape[-1] > 1 instead of {preds.shape[-1]}")


    if labels.shape[-1] == 1:
        labels_new = np.zeros(preds.shape)
        labels_new[np.arange(len(labels)), labels.reshape(-1).astype(int)] = 1
        labels = labels_new


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


def doner_specific_performance(pcl):
    """

    :param pcl: (dict) containing predictions and paths to those predictions
    :return: (dict) containing summary measures for doners
    """

    dsp = {}
    for key in pcl.keys():
        name = os.path.dirname(key)
        base = os.path.basename(key)
        if dsp.get(name, None) is None:
            dsp[name] = {}
            dsp[name]['predictions'] = list()
            dsp[name]['labels'] = list()
            dsp[name]['base'] = list()

        dsp[name]['base'].append(base)
        dsp[name]['predictions'].append(pcl[key]['predictions'])
        dsp[name]['labels'].append(pcl[key]['labels'])

    for name in dsp.keys():
        preds, labels = setup_matrix_for_single(dsp[name])
        metrics = get_simple_metrics(preds, labels)
        dsp[name]['metrics'] = metrics

    return dsp


def softmax(x):
    if x.shape[0] > 1:
        soft = np.exp(x)/np.tile(np.sum(np.exp(x), axis = 1)[:, None],x.shape[1])
    else:
        soft = np.exp(x)/np.sum(np.exp(x))
    return soft


def write_metrics(metrics):

    for key, val in metrics.items():
        print(f"{key} was {val}")

def keys_to_array(pcl, string, metric = 'accuracy'):

    vals = [val['metrics'][metric] for key, val in pcl.items() if string in key]

    return vals

def read_video_length(path):
    cap = cv2.VideoCapture(path)
    if cap.isOpened() == False:
        print("Something went wrong in the reading of the video")

    count = 0
    while True:
        ret, frame = cap.read()
        if ret:
            count += 1
        else:
            break
    return count

def compare_length_and_accuracy(dsp):
    """

    :param dsp: (dict) doner specific performance, with keys being the folders containing videos for each doner
    :return: (dict) copy of dsp with length of videos added
    """

    keys = list(dsp.keys())

    for key in keys:
        if os.path.isdir(key):
            dsp[key]['lengths'] = []
            for base in dsp[key]['base']:
                path = os.path.join(key, base)
                count = read_video_length(path)
                dsp[key]['lengths'].append(count)

    return dsp




if __name__ == '__main__':

    pcl = read_results(cfg = None, path =r'/scratch/s183993/outputs/SLOWFAST_8x8_R101_all_files_sample8/new_results_test.pkl')
    dsp =  doner_specific_performance(pcl)
    dsp = compare_length_and_accuracy(dsp)
    with open(r'/scratch/s183993/outputs/SLOWFAST_8x8_R101_all_files_sample8/dsp_test.pkl','wb') as handle:
        pickle.dump(dsp, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pcl = read_results(cfg=None, path=r'/scratch/s183993/outputs/SLOWFAST_8x8_R101_all_files_sample8/new_results_val.pkl')
    dsp = doner_specific_performance(pcl)
    dsp = compare_length_and_accuracy(dsp)
    with open(r'/scratch/s183993/outputs/SLOWFAST_8x8_R101_all_files_sample8/dsp_val.pkl','wb') as handle:
        pickle.dump(dsp, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # if False:
    #     preds, labels = setup_matrices(pcl)
    #     metrics = get_simple_metrics(preds, labels)
    #     write_metrics(metrics)
    #     fpr, tpr, roc_auc = calculate_roc_curve(preds, labels)
    #     plot_roc_curve(fpr, tpr,roc_auc)
    #     breakpoint()






