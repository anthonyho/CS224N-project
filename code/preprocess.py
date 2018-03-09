import os
import pickle
import numpy as np
import pandas as pd
import nltk
import utils
import vocab


id_unknown = 1
pad_token = u'<pad>'
glove_dir = '../data/glove'


def load_data(file_name, debug=False):
    if debug and isinstance(debug, bool):
        data = pd.read_csv(file_name, encoding='utf-8', nrows=6000)
    elif debug and isinstance(debug, int):
        data = pd.read_csv(file_name, encoding='utf-8', nrows=debug)
    else:
        data = pd.read_csv(file_name, encoding='utf-8')
    return data


def load_tokenized_comments(file_name):
    with open(file_name, 'rb') as f:
        tokenized_comments = pickle.load(f)
    return tokenized_comments


def _tokenize_single_string(string):
    tokens = [token.replace("``", '"').replace("''", '"').lower()
              for token in nltk.word_tokenize(string)]
    return tokens


def tokenize_df(df, column='comment_text'):
    list_list_tokens = []
    for index, item in df[column].iteritems():
        list_tokens = _tokenize_single_string(item)
        list_list_tokens.append(list_tokens)
    return list_list_tokens


def _uniformize_comment_length(list_tokens, length,
                               pad_token=pad_token):
    padded_list_tokens = list_tokens[:length]  # clip if too long
    mask = len(padded_list_tokens) * [True]
    while len(padded_list_tokens) < length:
        padded_list_tokens.append(pad_token)
        mask.append(False)
    return padded_list_tokens, mask


def pad_comments(list_list_tokens, length=None, **kwargs):
    padded_list_list_tokens = []
    masks = []
    for list_tokens in list_list_tokens:
        padded_list_tokens, mask = _uniformize_comment_length(list_tokens,
                                                              length, **kwargs)
        padded_list_list_tokens.append(padded_list_tokens)
        masks.append(mask)
    return padded_list_list_tokens, masks


def tokens_to_ids(list_list_tokens, word2id, id_unknown=id_unknown):
    return [[word2id.get(token, id_unknown) for token in list_tokens]
            for list_tokens in list_list_tokens]


def average_sentence_vectors(list_list_inds, emb_matrix):
    sentence_avgs = []
    for list_inds in list_list_inds:
        sentence_avgs.append(emb_matrix[list_inds].mean(axis=0))
    return np.array(sentence_avgs, dtype='float32')


def filter_labels(df_train, columns):
    return df_train[columns].values.astype('float32')


def get_glove(glove_dim, glove_file=None, glove_dir=glove_dir):
    if glove_file is None:
        glove_prefix = os.path.join(glove_dir, 'glove.6B.')
        glove_suffix = 'd.txt'
        glove_file = glove_prefix+str(glove_dim)+glove_suffix
    return vocab.get_glove(glove_file, glove_dim)


def split_train_dev(inputs, labels, masks=None,
                    fraction_dev=0.3, shuffle=True):
    '''
    Split data into train and dev sets

    Inputs:
    - inputs: list or numpy array (to be splitted across rows)
    - labels: list or numpy array (to be splitted across rows)
    - masks:  list or numpy array (to be splitted across rows)
    - fraction_dev: fraction of data to be held out for dev set
    - shuffle: bool to randomly shuffle data before splitting
    '''
    assert len(inputs) == len(labels), \
        "Inputs and labels must have equal dimensions!"
    if masks is not None:
        assert len(inputs) == len(masks), \
            "Inputs and labels must have equal dimensions!"
    n_data = len(inputs)
    ind = np.arange(n_data)
    if shuffle:
        np.random.shuffle(ind)
    boundary = int((1 - fraction_dev) * n_data)
    inputs_train = utils._get_items(inputs, ind[0:boundary])
    inputs_dev = utils._get_items(inputs, ind[boundary:])
    labels_train = utils._get_items(labels, ind[0:boundary])
    labels_dev = utils._get_items(labels, ind[boundary:])
    if masks is not None:
        masks_train = utils._get_items(masks, ind[0:boundary])
        masks_dev = utils._get_items(masks, ind[boundary:])
        return (inputs_train, labels_train, masks_train,
                inputs_dev, labels_dev, masks_dev)
    else:
        return (inputs_train, labels_train,
                inputs_dev, labels_dev)
