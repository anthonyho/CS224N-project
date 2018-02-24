from sklearn.metrics import roc_curve, roc_auc_score, \
    precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import utils

sns.set(style="white")
sns.set_palette("Paired", 12)
colors = sns.color_palette("Paired", 12)

metric_full = {'roc': 'ROC AUC',
               'prc': 'average precision'}


def evaluate_all(y_true, y_score,
                 metrics=['roc', 'prc'], names=None, plot=False):

    n_columns = y_true.shape[1]

    if not isinstance(metrics, list):
        metrics = [metrics]

    if names is None:
        names = ['column '+str(i+1) for i in range(n_columns)]

    results = {}
    for metric in metrics:
        results[metric] = {}
        scores = evaluate(y_true, y_score, metric=metric, average=False)
        for name, score in zip(names, scores):
            results[metric][name] = score
            print "{} of {} = {:.4f}".format(metric_full[metric], name, score)
        results[metric]['average'] = evaluate(y_true, y_score,
                                              metric=metric, average=True)
        print "Mean column-wise {} = {:.4f}".format(metric_full[metric], score)
    return results


def evaluate(y_true, y_score, metric='roc', average=True):
    '''Function to evaluate performance

    Inputs:
    - y_true: np.array of shape (n_samples, n_labels)
    - y_score: np.array of shape (n_samples, n_labels)
    - metric: 'roc' or 'prc'
    - average: return mean column-wise metric if true,
               return np.array of shape (n_labels) otherwise
    '''
    if average:
        if metric == 'roc':
            return roc_auc_score(y_true, y_score, average='macro')
        elif metric == 'prc':
            return average_precision_score(y_true, y_score, average='macro')
    else:
        if metric == 'roc':
            return roc_auc_score(y_true, y_score, average=None)
        elif metric == 'prc':
            return average_precision_score(y_true, y_score, average=None)


def plot_metric_curve(y_true, y_score, metric='roc', ax=None, **kwargs):
    '''Plot metric curve for a single label

    Inputs:
    - y_true: np.array of shape (n_samples, 1)
    - y_score: np.array of shape (n_samples, 1)
    - metric: 'roc' or 'prc'
    - ax: axes to plot in
    - kwargs: additional keyword arguments to pass to plt.plot / plt.step
    '''
    if ax is None:
        ax = plt.gca()

    if metric == 'roc':
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, linewidth=3, **kwargs)
        xlabel = 'False positive rate'
        ylabel = 'True positive rate'

    elif metric == 'prc':
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ax.step(recall, precision, linewidth=3, **kwargs)
        xlabel = 'Recall'
        ylabel = 'Precision'

    utils.setplotproperties(ax=ax, equal=True,
                            xlabel=xlabel, ylabel=ylabel,
                            legend=('label' in kwargs),
                            legendloc=(1.04, 0.8))
