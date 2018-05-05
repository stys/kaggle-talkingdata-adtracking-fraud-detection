# -*- coding: utf-8 -*-

import numpy as np


def reliability_curve(labels, predictions, nbins, sample_weights=None):
    """ Reliability curve for binary classification tasks
    Group samples into bins by value of predicted probability and compute empirical probability in each bin

    :param labels: true target values in {0, 1}
    :param predictions: predicted probabilities in [0.0, 1.0]
    :param nbins: number of bins
    :param sample_weights: use the same sample weights that were used for training
    :return:
    """

    labels = np.array(labels)
    predictions = np.array(predictions)
    weights = sample_weights if sample_weights is not None else np.ones(len(labels))

    assert len(labels) == len(predictions)
    assert len(labels) >= nbins

    ns = int(len(labels) / nbins)
    rem = len(labels) - ns

    sort_idx = np.argsort(predictions)
    count = np.zeros(nbins)
    avg_pred = np.zeros(nbins)
    avg_label = np.zeros(nbins)
    weight_total = np.zeros(nbins)

    jbin = 0
    for j, idx in enumerate(sort_idx):
        avg_pred[jbin] += predictions[idx]
        avg_label[jbin] += labels[idx] * weights[idx]
        weight_total[jbin] += weights[idx]
        count[jbin] += 1
        if rem > 0 and count[jbin] == ns + 1:
            jbin += 1
            rem -= 1
        elif rem == 0 and count[jbin] == ns:
            jbin += 1

    return avg_label / weight_total, avg_pred / count
