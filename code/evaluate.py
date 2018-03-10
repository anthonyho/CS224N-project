from sklearn.metrics import (roc_curve, precision_recall_curve,
                             roc_auc_score, average_precision_score)
import matplotlib.pyplot as plt
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
