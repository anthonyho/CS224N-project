from sklearn.metrics import roc_curve, roc_auc_score, \
    precision_recall_curve, average_precision_score


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
    if average:
        if metric == 'roc':
            return roc_auc_score(y_true, y_score, average='macro')
        if metric == 'prc':
            return average_precision_score(y_true, y_score, average='macro')
    else:
        if metric == 'roc':
            return roc_auc_score(y_true, y_score, average=None)
        if metric == 'prc':
            return average_precision_score(y_true, y_score, average=None)
