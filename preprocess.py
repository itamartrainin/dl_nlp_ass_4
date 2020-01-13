import torch
import numpy as np
import json
import re

data_prop = {
    'snli': {
        'train': {
            'fname': '../data/snli_1.0/snli_1.0/snli_1.0_train.jsonl',
            'length': 550152
        },
        'dev': {
            'fname': '../data/snli_1.0/snli_1.0/snli_1.0_dev.jsonl',
            'length': 10000
        },
        'test': {
            'fname': '../data/snli_1.0/snli_1.0/snli_1.0_test.jsonl',
            'length': 10000
        }
    },
    'multinli': {
        'train': {
            'fname': '../data/multinli_1.0/multinli_1.0/multinli_1.0_train.jsonl',
            'length': 392702
        },
        'dev_matched': {
            'fname': '../data/multinli_1.0/multinli_1.0/multinli_1.0_dev_matched.jsonl',
            'length': 10000
        },
        'dev_mismatched': {
            'fname': '../data/multinli_1.0/multinli_1.0/multinli_1.0_dev_mismatched.jsonl',
            'length': 10000
        }
    }
}

embds_data_path = '../data/glove.840B.300d/glove.840B.300d.txt'
embds_dim = 300
vocab_size= 2196017

LHS_max_sent_len = 401
RHS_max_sent_len = 70

tokenize_reg = re.compile(r'[ ()]')

null_string = '<NULL>'
pad_string = '-'

label_to_ix = {pad_string: 0, 'entailment': 1, 'neutral': 2, 'contradiction': 3}


def get_preprocessed_data(embds_fname, out_fname, saved_embds=True, saved_data=True):
    if saved_embds:
        embds_data = torch.load(embds_fname)
        embds = embds_data['embds']
        word_to_ix = embds_data['word_to_ix']
    else:
        embds, word_to_ix = read_pre_trained_embeds(vocab_size, embds_dim)
        torch.save({
            'embds': embds,
            'word_to_ix': word_to_ix
        }, embds_fname)

    # iterate over the different data-sets and extract the data from them
    data = {}
    if saved_data:
        data = torch.load(out_fname)
    else:
        for dname in data_prop:
            data[dname] = read_data(data_prop[dname], word_to_ix, label_to_ix)
        torch.save(data, out_fname)

    return embds, data, word_to_ix, label_to_ix


def read_pre_trained_embeds(vocab_size, embds_dim):
    with open(embds_data_path, 'r', encoding='utf-8') as f:

        embeds = torch.zeros(vocab_size, embds_dim)
        word_to_ix = {pad_string: 0, null_string: 1}

        i = 0
        for line in f.readlines():
            # all values separated by space
            # first value is the word, the rest are the embeddings
            content = line.split(' ')

            # add word to dictionary
            if content[0] not in word_to_ix:
                word_to_ix[content[0]] = len(word_to_ix)

            # add embed vector
            embeds_arr = np.array(content[1:])
            embeds_arr = embeds_arr.astype(float)
            embeds[i] = torch.tensor(embeds_arr)

            i += 1

    return embeds, word_to_ix


def read_data(props, word_to_ix, label_to_ix):

    ret = {}

    for ds_type in props:

        current_props = props[ds_type]

        data_size = current_props['length']
        LHS_word_ixs = torch.zeros(data_size, LHS_max_sent_len).long()
        RHS_word_ixs = torch.zeros(data_size, RHS_max_sent_len).long()
        LHS_lens = torch.zeros(data_size).long()
        RHS_lens = torch.zeros(data_size).long()
        labels = torch.zeros(data_size).long()

        with open(current_props['fname'], 'r', encoding='utf-8') as f:
            i = 0
            for line in f.readlines():
                line = json.loads(line)

                # Add gold label to labels
                labels[i] = label_to_ix[line['gold_label']]

                # Extract the tokens from the sentence
                LHS_tokens = token_ixs_in_sentence(line['sentence1_binary_parse'], word_to_ix)
                RHS_tokens = token_ixs_in_sentence(line['sentence2_binary_parse'], word_to_ix)

                LHS_word_ixs[i][:LHS_tokens.size(0)] = LHS_tokens
                RHS_word_ixs[i][:RHS_tokens.size(0)] = RHS_tokens

                LHS_lens[i] = LHS_tokens.size(0)
                RHS_lens[i] = RHS_tokens.size(0)

                i += 1

        ret[ds_type] = (LHS_word_ixs, RHS_word_ixs, LHS_lens, RHS_lens, labels)

    return ret


def token_ixs_in_sentence(sentence, word_to_ix):
    tokens = tokenize_reg.split(sentence)

    # filter blanks
    tokens = list(filter(lambda x: x != '', tokens))

    # convert word to indexes
    token_ixs = list(map(lambda x: word_to_ix[x] if x in word_to_ix else word_to_ix[null_string], tokens))

    return torch.tensor(token_ixs).long()


if __name__ == '__main__':
    preprocessed_data_fname = '../data/preprocessed_data'
    embds_fname = '../data/embds'
    get_preprocessed_data(embds_fname, preprocessed_data_fname, True, False)
