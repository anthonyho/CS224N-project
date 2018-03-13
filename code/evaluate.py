import numpy as np
from sklearn.metrics import (roc_curve, precision_recall_curve,
                             roc_auc_score, average_precision_score)
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import logging
import utils


sns.set(style="white")
sns.set_palette("Paired", 12)
colors = sns.color_palette("Paired", 12)

metric_long = {'roc': 'ROC AUC',
               'prc': 'average precision'}

datasets = ['train', 'dev', 'test']

dataset_linestyles = {'train': '-',
                      'dev': '--',
                      'test': ':'}


def evaluate_full(y_dict, metric='roc', names=None,
                  plot=True, save_prefix=None):
    '''
    Detailed evaluation of multilabel classification

    Inputs:
    - y_dict: dict with keys in {'train', 'dev', 'test'} and
              values = (y_true, y_prob) where y_true and y_prob
              are np.array of shape (n_samples, n_labels)
    - metric: 'roc' or 'prc'
    - names: list of names for each label (e.g. ['toxic', 'obscene', 'insult'])
    - plot: bool to plot ROC/PRC
    - save_prefix: file path to save the figure (no extension)

    Return:
    dict['average'|label]['train'|'dev'|'test']
    '''
    # Get current root logger
    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s -- %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

    # Get datasets in y_dict
    curr_datasets = [dataset for dataset in datasets if dataset in y_dict]

    # Create default names if not provided
    n_columns = y_dict[curr_datasets[0]][0].shape[1]
    if names is None:
        names = ['column '+str(i+1) for i in range(n_columns)]

    # Evaluate all results
    results = {}
    results['average'] = {}
    # Evaluate mean column-wise result
    for dataset in curr_datasets:
        y_true = y_dict[dataset][0]
        y_prob = y_dict[dataset][1]
        score = evaluate(y_true, y_prob, metric=metric, average=True)
        results['average'][dataset] = score
        message = "Mean column-wise {} - {} = {:.4f}"
        logger.info(message.format(metric_long[metric], dataset, score))
    # Initialize plot if requested
    if plot:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111)
    # Evaluate for each label
    for i, name in enumerate(names):
        results[name] = {}
        for dataset in curr_datasets:
            y_true = y_dict[dataset][0][:, i]
            y_prob = y_dict[dataset][1][:, i]
            score = evaluate(y_true, y_prob, metric=metric)
            results[name][dataset] = score
            message = "{} of {} - {} = {:.4f}"
            logger.info(message.format(metric_long[metric], name,
                                       dataset, score))
            if plot:
                label = name + ' - ' + dataset
                color = colors[2 * i + 1]
                plot_metric_curve(y_true, y_prob, metric=metric, ax=ax,
                                  label=label, color=color,
                                  linestyle=dataset_linestyles[dataset])
    # Save fig
    if save_prefix:
        plt.savefig(save_prefix+'_'+metric+'.png', bbox_inches='tight')
        plt.savefig(save_prefix+'_'+metric+'.eps', bbox_inches='tight')

    return results


def evaluate(y_true, y_prob, metric='roc', average=True):
    '''Evaluate performance of multilabel classification

    Inputs:
    - y_true: np.array of shape (n_samples, n_labels)
    - y_prob: np.array of shape (n_samples, n_labels)
    - metric: 'roc' or 'prc'
    - average: return mean column-wise metric if true,
               return np.array of shape (n_labels) otherwise
    '''
    if average:
        if metric == 'roc':
            return roc_auc_score(y_true, y_prob, average='macro')
        elif metric == 'prc':
            return average_precision_score(y_true, y_prob, average='macro')
    else:
        if metric == 'roc':
            return roc_auc_score(y_true, y_prob, average=None)
        elif metric == 'prc':
            return average_precision_score(y_true, y_prob, average=None)


def plot_metric_curve(y_true, y_prob, metric='roc', ax=None, **kwargs):
    '''Plot metric curve for a single label

    Inputs:
    - y_true: np.array of shape (n_samples, 1)
    - y_prob: np.array of shape (n_samples, 1)
    - metric: 'roc' or 'prc'
    - ax: axes to plot in
    - kwargs: additional keyword arguments to pass to plt.plot / plt.step
    '''
    if ax is None:
        ax = plt.gca()

    if metric == 'roc':
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, linewidth=3, **kwargs)
        xlabel = 'False positive rate'
        ylabel = 'True positive rate'

    elif metric == 'prc':
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ax.step(recall, precision, linewidth=3, **kwargs)
        xlabel = 'Recall'
        ylabel = 'Precision'

    utils.setplotproperties(ax=ax, equal=True,
                            xlabel=xlabel, ylabel=ylabel,
                            legend=('label' in kwargs),
                            legendloc=(1.04, 0))


def plot_loss(list_loss_train, list_loss_dev=None,
              save_prefix=None):
    '''
    Plot loss over epoch

    Inputs:
    - save_prefix: file path to save the figure (no extension)
    '''
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    ax.plot(list_loss_train, linewidth=3, color=colors[3], label='train')
    if list_loss_dev is not None:
        ax.plot(list_loss_dev, linewidth=3, color=colors[5], label='dev')
    utils.setplotproperties(ax=ax, xlabel='Epoch', ylabel='Loss',
                            legend=True, legendloc=1)

    if save_prefix:
        plt.savefig(save_prefix+'_loss'+'.png', bbox_inches='tight')
        plt.savefig(save_prefix+'_loss'+'.eps', bbox_inches='tight')


def plot_score(list_score_train, list_score_dev=None, metric='roc',
               save_prefix=None):
    '''
    Plot performance score over epoch

    Inputs:
    - save_prefix: file path to save the figure (no extension)
    '''
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    ax.plot(list_score_train, linewidth=3, color=colors[7], label='train')
    if list_score_dev is not None:
        ax.plot(list_score_dev, linewidth=3, color=colors[9], label='dev')
    utils.setplotproperties(ax=ax, xlabel='Epoch', ylabel=metric_long[metric],
                            legend=True, legendloc=4)

    if save_prefix:
        plt.savefig(save_prefix+'_score'+'.png', bbox_inches='tight')
        plt.savefig(save_prefix+'_score'+'.eps', bbox_inches='tight')


def _break_line(sentence, n_char_tokens, alphas, max_width):
    '''
    sentence - list of tokens of length n_tokens
    n_char_tokens - list of int of length n_tokens, with each element
                    indicating the length of the token
    alphas - np.array of shape (n_tokens, )
    max_width - int
    '''
    # Just in case if any single word is too long
    width = max(max(n_char_tokens), max_width)
    # Packing tokens and alphas into lines/rows
    paragraph = []
    line = []
    matrix = []
    row = []
    n_char_line = 0
    for token, n_char_token, alpha in zip(sentence, n_char_tokens, alphas):
        if n_char_line + n_char_token <= width:
            line.extend(list(token))
            row.extend([alpha] * n_char_token)
            n_char_line += n_char_token
        else:
            paragraph.append(line)
            matrix.append(row)
            line = list(token)
            row = [alpha] * n_char_token
            n_char_line = n_char_token
    paragraph.append(line)
    matrix.append(row)
    # Trimming the ends if there are excess spaces left
    width = max([len(line) for line in paragraph])
    # Padding with spaces/zeros until uniform shapes
    for line in paragraph:
        line.extend(' ' * (width - len(line)))
    for row in matrix:
        row.extend([0] * (width - len(row)))
    return paragraph, np.array(matrix)


def highlight_sentence(sentence, alphas, masks=None, max_width=80):
    '''
    sentence - list of tokens of length n_tokens
    alphas - np.array of shape (n_tokens, )
    masks - list of bool of length n_tokens
    max_width - int
    '''
    # Truncate sentence and alphas according to masks
    if masks is not None:
        sentence = [token for token, mask in zip(sentence, masks) if mask]
        alphas = alphas[masks]
    # Add space right after each token
    sentence = [token+' ' for token in sentence]
    # Count number of characters per token
    n_char_tokens = [len(token) for token in sentence]
    # Format into paragraph
    paragraph, matrix = _break_line(sentence, n_char_tokens, alphas, max_width)
    (height, width) = matrix.shape

    # Define constants
    unit_x = 0.3227
    unit_y = 0.6
    offset_x = 0.5
    offset_y = 0.12

    # Make heatmap
    fig_width = unit_x * width
    fig_height = unit_y * height
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111)
    ax.axis('off')
    font = FontProperties()
    font.set_family('monospace')
    cmap = sns.cubehelix_palette(8, as_cmap=True)
    mesh = ax.pcolormesh(matrix[::-1], cmap=cmap)
    mesh.update_scalarmappable()

    # Add characters
    for i, line in enumerate(paragraph):
        for j, char in enumerate(line):
            x_pos = j + offset_x
            y_pos = (height - i - 1) + offset_y
            color = mesh.get_facecolors()[(height - i - 1) * width + j]
            l = sns.utils.relative_luminance(color)
            text_color = '.15' if l > .408 else 'w'
            ax.text(x_pos, y_pos, char,
                    fontproperties=font, size=30, color=text_color,
                    horizontalalignment='left', verticalalignment='baseline')
