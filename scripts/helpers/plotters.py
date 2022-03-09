from sklearn import metrics
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)


def plot_wrapper(plot_object, title, subtitle, xlabel, ylabel, xtick_rotation=None):
    """
    Simple function to set title, labels etc. of a seaborn plot. Helps with readability.
    The function prints the plot.

    Arguments
    ---------
    :param plot_object:    a seaborn plot object
    :param title:          str, sets a bold title
    :param subtitle:       str, sets a subtitle
    :param xlabel:         str, replaces xlabel
    :param ylabel:         str, replaces ylabel
    :param xtick_rotation: int, rotates xticks

    Returns
    -------
    :return None:          the function prints a plot
    """
    g = plot_object

    # Title, Subtitle and Axis
    g.text(x=0.5,
           y=1.06,
           s=title,
           fontsize=10, weight='bold', ha='center', va='bottom', transform=g.transAxes)
    g.text(x=0.5,
           y=1.01,
           s=subtitle,
           fontsize=10, alpha=0.75, ha='center', va='bottom', transform=g.transAxes)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xtick_rotation is not None:
        plt.xticks(rotation=xtick_rotation)
    plt.show()


def evaluate_binary(evaluation_folder, title, y_true, y_pred):
    plot_curves(evaluation_folder, title, y_true, y_pred)
    plot_confusion_matrix(evaluation_folder, title, y_true, y_pred, False)
    plot_confusion_matrix(evaluation_folder, title, y_true, y_pred, True)
    plot_prediction_distribution(evaluation_folder, title, y_true, y_pred)


def plot_curves(experiment_path, title, y_true, y_pred):
    # calculate sample counts
    n_total = len(y_true)
    n_postive = (y_true == 1).sum()
    n_negative = (y_true == 0).sum()

    # calculate inverse label & preiction for negative label
    not_y_true = y_true.apply(lambda x: 1 - x)
    not_y_prediction = y_pred.apply(lambda x: 1 - x)

    # roc / pr curve on positive label
    fpr, tpr, roc_thresholds = metrics.roc_curve(y_true, y_pred)
    precision, recall, prc_thresholds = metrics.precision_recall_curve(y_true, y_pred)

    # roc / pr curve on negative label
    fnr, tnr, negative_roc_thresholds = metrics.roc_curve(not_y_true, not_y_prediction)
    negative_precision, negative_recall, negative_prc_thresholds = metrics.precision_recall_curve(not_y_true,
                                                                                                  not_y_prediction)

    plt.rcParams['axes.grid'] = True
    plt.rcParams['font.size'] = 12

    fig, ax = plt.subplots(2, 2, figsize=(14, 14))

    fig.suptitle('%s (Samples=%d, Pos=%d (%.2f%%),  Neg=%d (%.2f%%))' % (
    title, n_total, n_postive, 100 * n_postive / n_total, n_negative, 100 * n_negative / n_total), fontsize=16)

    major_ticks = np.arange(0, 1.1, 0.1)
    plt.setp(ax, xticks=major_ticks, yticks=major_ticks)

    plt.setp(ax, xlim=[0, 1], ylim=[0, 1])

    # plot curves for positive label
    ax[0, 0].plot(fpr, tpr)
    ax[0, 0].set_xlabel('FPR')
    ax[0, 0].set_ylabel('TPR')
    ax[0, 0].text(0.05, 0.05, 'AUC=%.2f' % metrics.auc(fpr, tpr), ha="left", va="center", color="black")

    roc_threshold_indices = np.linspace(0, len(roc_thresholds) - 1, 17)
    for threshold_idx in roc_threshold_indices[1:-1]:
        threshold_idx = int(round(threshold_idx))
        ax[0, 0].plot([fpr[threshold_idx]], [tpr[threshold_idx]], marker='o', markersize=3, color="red")
        ax[0, 0].text(fpr[threshold_idx] + 0.01, tpr[threshold_idx], '%.2f' % roc_thresholds[threshold_idx], fontsize=9,
                      verticalalignment='center')

    ax[0, 1].plot(recall, precision)
    ax[0, 1].set_xlabel('Recall')
    ax[0, 1].set_ylabel('Precision')
    ax[0, 1].text(0.05, 0.05, 'AUC=%.2f' % metrics.auc(recall, precision), ha="left", va="center", color="black")
    ax[0, 1].text(0.95, 0.05, 'Positive Label' % metrics.auc(fnr, tnr), ha="right", va="center", color="black")

    prc_threshold_indices = np.linspace(0, len(prc_thresholds) - 1, 17)
    for threshold_idx in prc_threshold_indices[1:-1]:
        threshold_idx = int(round(threshold_idx))
        ax[0, 1].plot([recall[threshold_idx]], [precision[threshold_idx]], marker='o', markersize=3, color="red")
        ax[0, 1].text(recall[threshold_idx], precision[threshold_idx], '%.2f' % prc_thresholds[threshold_idx],
                      fontsize=9, verticalalignment='bottom')

    # plot curves for negative label
    ax[1, 0].plot(fnr, tnr)
    ax[1, 0].set_xlabel('FNR')
    ax[1, 0].set_ylabel('TNR')
    ax[1, 0].text(0.05, 0.05, 'AUC=%.2f' % metrics.auc(fnr, tnr), ha="left", va="center", color="black")

    negative_roc_threshold_indices = np.linspace(0, len(negative_roc_thresholds) - 1, 17)
    for threshold_idx in negative_roc_threshold_indices[1:-1]:
        threshold_idx = int(round(threshold_idx))
        ax[1, 0].plot([fnr[threshold_idx]], [tnr[threshold_idx]], marker='o', markersize=3, color="red")
        ax[1, 0].text(fnr[threshold_idx] + 0.01, tnr[threshold_idx],
                      '%.2f' % (1 - negative_roc_thresholds[threshold_idx]), fontsize=9, verticalalignment='center')

    ax[1, 1].plot(negative_recall, negative_precision)
    ax[1, 1].set_xlabel('Recall')
    ax[1, 1].set_ylabel('Precision')
    ax[1, 1].text(0.05, 0.05, 'AUC=%.2f' % metrics.auc(negative_recall, negative_precision), ha="left", va="center",
                  color="black")
    ax[1, 1].text(0.95, 0.05, 'Negative Label' % metrics.auc(fnr, tnr), ha="right", va="center", color="black")

    negative_prc_threshold_indices = np.linspace(0, len(negative_prc_thresholds) - 1, 17)
    for threshold_idx in negative_prc_threshold_indices[1:-1]:
        threshold_idx = int(round(threshold_idx))
        ax[1, 1].plot([negative_recall[threshold_idx]], [negative_precision[threshold_idx]], marker='o', markersize=3,
                      color="red")
        ax[1, 1].text(negative_recall[threshold_idx], negative_precision[threshold_idx],
                      '%.2f' % (1 - negative_prc_thresholds[threshold_idx]), fontsize=9, verticalalignment='bottom')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(experiment_path + '/' + title)


def plot_confusion_matrix(experiment_path, title, y_true, y_pred, is_normalized=False):
    # calculate sample counts
    n_total = len(y_true)
    n_postive = (y_true == 1).sum()
    n_negative = (y_true == 0).sum()

    # general plot parameters
    plt.rcParams['axes.grid'] = False
    plt.rcParams['font.size'] = 16

    # create 3x3 plots
    fig, ax = plt.subplots(3, 3, figsize=(20, 20))

    # reduce spacing between subplots
    fig.subplots_adjust(hspace=0, wspace=0.1)

    fig.suptitle('%s (Samples=%d, Pos=%d (%.2f%%),  Neg=%d (%.2f%%))' % (
    title, n_total, n_postive, 100 * n_postive / n_total, n_negative, 100 * n_negative / n_total), fontsize=24, y=0.92)

    # add x label and y label for the middle plot (horizontally and vertically)
    ax[1, 0].set_ylabel('Truth')
    ax[2, 1].set_xlabel('Prediction')

    # add confusion matrix plot for each threshold in '0.1' steps
    for th_idx, threshold in enumerate(np.arange(0.1, 1, 0.1)):
        # calculate position in 3x3 grid
        plot_x_idx = int(th_idx / 3)
        plot_y_idx = th_idx % 3

        # apply threshold and calculate metrics
        prediction = y_pred.apply(lambda x: 1.0 if x > threshold else 0.0)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, prediction).ravel()
        n_all = tn + fp + fn + tp
        n_label_pos = tp + fn
        n_label_neg = tn + fp

        # max limit for colorbar is the number of samples
        vmax = n_all

        # if desired, normalize metrics and adjust colorbar
        if is_normalized:
            tp = tp / n_label_pos
            fp = fp / n_label_neg
            tn = tn / n_label_neg
            fn = fn / n_label_pos
            vmax = 1

        # plot confusion
        im = ax[plot_x_idx, plot_y_idx].imshow([[tn, fp], [fn, tp]], vmin=0, vmax=vmax, cmap='Blues')

        # remove all ticks
        ax[plot_x_idx, plot_y_idx].set_xticks([])
        ax[plot_x_idx, plot_y_idx].set_yticks([])

        # Loop over data dimensions and create text annotations.
        ax[plot_x_idx, plot_y_idx].text(0, 0, '%.2f' % tn if is_normalized else tn, ha="center", va="center",
                                        color="black")
        ax[plot_x_idx, plot_y_idx].text(1, 0, '%.2f' % fp if is_normalized else fp, ha="center", va="center",
                                        color="black")
        ax[plot_x_idx, plot_y_idx].text(0, 1, '%.2f' % fn if is_normalized else fn, ha="center", va="center",
                                        color="black")
        ax[plot_x_idx, plot_y_idx].text(1, 1, '%.2f' % tp if is_normalized else tp, ha="center", va="center",
                                        color="black")

        # set subplot tile with threshold
        ax[plot_x_idx, plot_y_idx].set_title('Threshold=%.2f' % threshold)

    # add ticks and labels only completely left and at the very bottom
    for position in range(0, 3):
        # set ticks
        ax[2, position].set_xticks([0, 1])
        ax[position, 0].set_yticks([0, 1])
        ax[position, 0].set_ylim(1.5, -0.5)

        # set tick labels
        ax[2, position].set_xticklabels(['False', 'True'])
        ax[position, 0].set_yticklabels(['False', 'True'])

    # add one colorbar for all plots
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.14, 0.05, 0.72])
    fig.colorbar(im, cax=cbar_ax)

    plt.savefig(experiment_path + '/%s_confusion_matrix_%s_threshold_%d' % (
    title, 'normalized' if is_normalized else 'unnormalized', int(100 * threshold)))


def cumulative(bins, reverse=False):
    if reverse:
        bins = bins[::-1]

    result = [0]
    for binValue in bins:
        result.append(result[-1] + binValue)

    if reverse:
        result = result[::-1]
        maxValue = result[0]
    else:
        maxValue = result[-1]

    return result / maxValue


def plot_prediction_distribution(experiment_path, title, y_true, y_pred):
    fig, ax1 = plt.subplots(figsize=(14, 8))

    positive_label_prediction = y_pred[y_true == 1]
    negative_label_prediction = y_pred[y_true == 0]

    n_total = len(y_true)
    n_postive = len(positive_label_prediction)
    n_negative = len(negative_label_prediction)

    fig.suptitle('%s (Samples=%d, Pos=%d (%.2f%%),  Neg=%d (%.2f%%))' % (
    title, n_total, n_postive, 100 * n_postive / n_total, n_negative, 100 * n_negative / n_total), fontsize=16)

    binValues, binLimits, _ = plt.hist([positive_label_prediction, negative_label_prediction], range=(0, 1), bins=100,
                                       color=['green', 'red'])

    ax1.tick_params(axis='y')

    data_no_paper_true_cumulative = cumulative(binValues[1])
    data_no_paper_true_cumulative_reverse = cumulative(binValues[1], reverse=True)

    data_no_paper_false_cumulative = cumulative(binValues[0])
    data_no_paper_false_cumulative_reverse = cumulative(binValues[0], reverse=True)

    ax1.set_xlabel('Predicted Probability')
    ax1.set_ylabel('Occurences')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Density')  # we already handled the x-label with ax1

    ax1.set_xlim(xmin=0, xmax=1)
    ax2.set_xlim(xmin=0, xmax=1)
    ax2.set_ylim(ymin=0, ymax=1.1)

    ax2.xaxis.set_major_locator(MultipleLocator(0.1))
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.xaxis.set_minor_locator(MultipleLocator(0.02))

    a = ax2.plot(binLimits, data_no_paper_true_cumulative, 'r-', alpha=0.5, label='Negative (left CDF)')
    b = ax2.plot(binLimits, data_no_paper_true_cumulative_reverse, 'r--', alpha=0.5, label='Negative (right CDF)')

    c = ax2.plot(binLimits, data_no_paper_false_cumulative, 'g-', alpha=0.5, label='Positive (left CDF)')
    d = ax2.plot(binLimits, data_no_paper_false_cumulative_reverse, 'g--', alpha=0.5, label='Positive (right CDF)')

    ax2.tick_params(axis='y')

    ax2.legend(ncol=4, bbox_to_anchor=(0.5, 1), loc='lower center', fontsize='small')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.grid(b=True, which='both', axis='both')

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(experiment_path + '/%s_prediction_distribution' % (title))
