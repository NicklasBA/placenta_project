
import os
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={'figure.facecolor':'white'})



def read_stdout(path, max_epocs = 195):
    """

    :param path: Path to stdout file
    :return: training and validation errors over epochs
    """

    val_score = []
    num_epochs = 0
    train_score = []
    val_epochs = []

    with open(path) as f:
        lines = f.readlines()

        for idx, line in enumerate(lines):
            ll = line.split(" ")
            if len(ll) >= 10:
                if 'val_epoch' in ll[9]:
                    err_idx = [idx  for idx, i in enumerate(ll) if 'top1_err' in i and 'min' not in i][0]
                    epoch_idx = [idx for idx, i in enumerate(ll) if 'epoch' in i and "_" not in i][0]
                    val_score.append(ll[err_idx + 1])
                    ep = ll[epoch_idx+1].split("/")[0]
                    val_epochs.append(ep)

                elif 'train_epoch' in ll[9]:

                    err_idx = [idx for idx, i in enumerate(ll) if 'top1_err' in i and 'min' not in i][0]

                    train_score.append(ll[err_idx + 1])
                    num_epochs += 1
            if num_epochs >= max_epocs:
                break


    return val_score, train_score, num_epochs, val_epochs


def plot_val_and_train(val_score, train_score, val_epochs):

    scores = pd.DataFrame()

    n_val = [None for _ in range(len(train_score))]
    for idx, i in enumerate(val_epochs[:-1]):

        n_val[i] = val_score[idx]
    n_val[-1] = val_score[-1]

    scores['Train Error'] = train_score
    scores['Val Error'] = n_val

    sns.lineplot(data = scores)
    sns.despine(top = True, bottom=True, left = True, right = True)
    plt.savefig(r'C:\Users\ptrkm\Action Classification\val_vs_train_error_org.eps', bbox_inches = "tight")


def clean(list_of_inputs):
    nvals = []
    for item in list_of_inputs:
        item = item.split(",")
        item = np.float32(item[0])
        nvals.append(item)

    return nvals

def clean_epochs(list_of_inputs):
    nvals = []
    for item in list_of_inputs:
        try:
            if "'" in item:
                item = item.replace("'", "")
            if '"' in item:
                item = item.replace('"', "")
            if "." in item:
                item = item.replace(".", "")
            nvals.append(np.int32(item))
        except:
            breakpoint()


    return nvals


if __name__ == '__main__':

    path = r'stdout_org.log'

    val_score, train_score, num_epochs, val_epochs = read_stdout(path, 200)
    val_score = clean(val_score)
    train_score = clean(train_score)
    val_epochs = clean_epochs(val_epochs)

    plot_val_and_train(val_score,train_score, val_epochs)

    breakpoint()


