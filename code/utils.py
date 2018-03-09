import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def softmax(x):
    if len(x.shape) > 1:
        x = np.exp(x - np.max(x, axis=1, keep_dims=True))
        x = x / np.sum(x, axis=1, keep_dims=True)
    else:
        x = np.exp(x - np.max(x))
        x = x / np.sum(x)
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def minibatch(batch_size, inputs, masks=None, labels=None, shuffle=True):
    '''
    Return generator for minibatching

    Inputs:
    - batch_size: int
    - inputs: list or numpy array (to be batched across rows)
    - masks: list or numpy array (to be batched across rows)
    - labels: list or numpy array (to be batched across rows)
    - shuffle: bool to randomly shuffle data before minibatching

    Return:
    - generator of inputs_batch if labels=None
    - generator of (inputs_batch, masks_batch) if masks but no labels
    - generator of (inputs_batch, labels_batch) if labels but no mask
    - generator of (inputs_batch, mask_batch, labels_batch) if all 3.
    '''
    if masks is not None:
        assert len(inputs) == len(masks), \
            'Inputs and masks must have equal dimensions!'
    if labels is not None:
        assert len(inputs) == len(labels), \
            'Inputs and labels must have equal dimensions!'
    n_data = len(inputs)
    ind = np.arange(n_data)
    if shuffle:
        np.random.shuffle(ind)
    for i in np.arange(0, n_data, batch_size):
        inputs_batch = _get_items(inputs, ind[i:i+batch_size])
        if masks is None and labels is None:
            yield inputs_batch
        elif masks is None and labels is not None:
            labels_batch = _get_items(labels, ind[i:i+batch_size])
            yield (inputs_batch, labels_batch)
        elif masks is not None and labels is None:
            masks_batch = _get_items(masks, ind[i:i+batch_size])
            yield (inputs_batch, masks_batch)
        else:
            masks_batch = _get_items(masks, ind[i:i+batch_size])
            labels_batch = _get_items(labels, ind[i:i+batch_size])
            yield (inputs_batch, masks_batch, labels_batch)


def _get_items(data, ind):
    '''Helper function to access items from a list of indices
    depending on data type'''
    if isinstance(data, list):
        return [data[i] for i in ind]
    else:
        return data[ind]


def y_prob_to_df(y_prob, comment_id, label_names):
    '''
    Return a Pandas dataframe of y_prob with id and column labels

    Inputs:
    - y_prob: np.array of shape (n_samples, n_labels)
    - comment_id: pandas series or list of length n_samples
    - label_names: list of label names of length n_labels
    '''
    comment_id = pd.Series(comment_id, name='id')
    y_prob_df = pd.DataFrame(y_prob, columns=label_names)
    y_prob_df = pd.concat([comment_id, y_prob_df], axis=1)
    return y_prob_df.fillna(0.5)  # <- quick hack, to be fixed


def setplotproperties(fig=None, ax=None, figsize=None,
                      suptitle=None, title=None,
                      legend=None, legendloc=1, legend_bbox=None,
                      legendwidth=2.5, legendbox=None,
                      xlabel=None, ylabel=None, xlim=None, ylim=None,
                      scix=False, sciy=False,
                      scilimitsx=(-3, 3), scilimitsy=(-3, 3),
                      logx=False, logy=False, majorgrid=None, minorgrid=None,
                      borderwidth=2.5, tight=True, pad=1.6,
                      fontsize=None, legendfontsize=20, tickfontsize=20,
                      labelfontsize=20, titlefontsize=18, suptitlefontsize=20,
                      xticklabelrot=None, yticklabelrot=None,
                      equal=False, symmetric=False):
    '''Convenient tool to set properties of a plot in a single command'''
    # Get figure and axis handles
    if not fig:
        fig = plt.gcf()
    if not ax:
        ax = plt.gca()

    # Set background color to white
    fig.patch.set_facecolor('w')

    # Define figure size if provided
    if figsize:
        fig.set_size_inches(figsize, forward=True)

    # Set titles if provided
    if suptitle is not None:
        if fontsize is None:
            fig.suptitle(suptitle, fontsize=suptitlefontsize, y=0.99)
        else:
            fig.suptitle(suptitle, fontsize=fontsize, y=0.99)
    if title is not None:
        ax.set_title(title, y=1.02)
    # Show legend if requested
    if legend:
        legend = ax.legend(bbox_to_anchor=legend_bbox, loc=legendloc,
                           numpoints=1, fontsize=legendfontsize,
                           frameon=legendbox)
        legend.get_frame().set_linewidth(legendwidth)
    # Set x and y labels if provided
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    # Set x and y limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    # Apply scientific notation to x and y tick marks if requested
    if scix:
        ax.ticklabel_format(axis='x', style='sci', scilimits=scilimitsx)
    if sciy:
        ax.ticklabel_format(axis='y', style='sci', scilimits=scilimitsy)
    # Change axis to log scale if requested
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    # Set major and minor grid in plot
    if majorgrid is not None:
        ax.grid(b=majorgrid, which='major')
    if minorgrid is not None:
        ax.grid(b=minorgrid, which='minor')

    # Rotate x and y tick labels if requested
    if xticklabelrot is not None:
        xticklabels = ax.get_xticklabels()
        for ticklabel in xticklabels:
            ticklabel.set_rotation(xticklabelrot)
    if yticklabelrot is not None:
        yticklabels = ax.get_yticklabels()
        for ticklabel in yticklabels:
            ticklabel.set_rotation(yticklabelrot)

    # Set borderwidth (not visible if using seaborn default theme)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(borderwidth)

    # Set individual fontsizes if fontsize is not specified
    if fontsize is None:
        plt.setp(ax.get_xticklabels(), fontsize=tickfontsize)
        plt.setp(ax.get_yticklabels(), fontsize=tickfontsize)
        ax.xaxis.label.set_fontsize(labelfontsize)
        ax.yaxis.label.set_fontsize(labelfontsize)
        ax.title.set_fontsize(titlefontsize)
    # Set all fontsizes to fontsize if fontsize is specified
    else:
        plt.setp(ax.get_xticklabels(), fontsize=fontsize)
        plt.setp(ax.get_yticklabels(), fontsize=fontsize)
        ax.xaxis.label.set_fontsize(fontsize)
        ax.yaxis.label.set_fontsize(fontsize)
        ax.title.set_fontsize(fontsize)

    # Set tight figure and padding
    if tight:
        fig.tight_layout(pad=pad)

    # Set equal aspect
    if equal:
        ax.set_aspect('equal', adjustable='box')

    # Set symmetric axis limits
    if symmetric:
        xlim_abs = max(abs(i) for i in ax.get_xlim())
        ylim_abs = max(abs(i) for i in ax.get_ylim())
        ax.set_xlim((-xlim_abs, xlim_abs))
        ax.set_ylim((-ylim_abs, ylim_abs))
