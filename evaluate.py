from sklearn.metrics import roc_curve, roc_auc_score, \
    precision_recall_curve, average_precision_score


def evaluate(y_true, y_score,
             metrics=['roc', 'prc'], names=None): # add plot

    n_columns = y_true.shape[1]

    if names is None:
        names = ['Column '+str(i+1) for i in range(n_columns)]

    results = {}
    for metric in metrics:
        results[metric] = {}
        scores = compute_score(y_true, y_score, metric=metric, average=False)
        for name, score in zip(names, scores):
            results[metric][name] = score
        results[metric]['average'] = compute_score(y_true, y_score,
                                                   metric=metric, average=True)
    return results


def compute_score(y_true, y_score, metric='roc', average=True):
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

