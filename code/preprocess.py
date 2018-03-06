import pickle
import numpy as np
import pandas as pd
import nltk
import utils
import vocab


id_unknown = 1


def load_data(file_name):
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

def _uniformize_list_tokens_length(list_tokens, target_length,zero_token_id=u'<UNK>'):
    padded_list_tokens = list_tokens[:target_length] #clip if too long
    mask = len(padded_list_tokens) * [True]
    while len(padded_list_tokens) < target_length:
        padded_list_tokens.append(zero_token_id)
        mask.append(False)
    return padded_list_tokens, mask

def tokenize_df(df, column='comment_text',target_length=None,**kwargs):
    list_list_tokens = []
    masks = []
    for index, item in df[column].iteritems():
        list_tokens = _tokenize_single_string(item)
        if target_length is not None:
            padded_list_tokens, mask = _uniformize_list_tokens_length(list_tokens,target_length,**kwargs)
            list_list_tokens.append(padded_list_tokens)
            masks.append(mask)
        else: 
            list_list_tokens.append(list_tokens)
    if target_length is not None:
        return list_list_tokens, masks
    else:
        return list_list_tokens

def tokens_to_ids(list_list_tokens, word2id):
    return [[word2id.get(token, id_unknown) for token in list_tokens]
            for list_tokens in list_list_tokens]


def average_sentence_vectors(list_list_inds, emb_matrix):
    sentence_avgs = []
    for list_inds in list_list_inds:
        sentence_avgs.append(emb_matrix[list_inds].mean(axis=0))
    return np.array(sentence_avgs, dtype='float32')


def filter_labels(df_train, columns):
    return df_train[columns].values.astype('float32')


def get_glove(glove_dim, glove_file=None):
    if glove_file is None:
        glove_prefix = '../data/glove/glove.6B.'
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
    =- 
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
    labels_train = utils._get_items(labels, ind[0:boundary])
    if masks is not None:
        masks_train = utils._get_items(masks, ind[0:boundary])
    inputs_dev = utils._get_items(inputs, ind[boundary:])
    labels_dev = utils._get_items(labels, ind[boundary:])
    if masks is not None:
        masks_dev = utils._get_items(masks, ind[boundary:])
    if masks is not None:
        return inputs_train, labels_train, masks_train, inputs_dev, labels_dev, masks_dev
    else:
        return inputs_train, labels_train, inputs_dev, labels_dev
