import torch
import json

snli = {
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
}

LHS_max_sent_len = 82
RHS_max_sent_len = 62



tag_ix = {'entailment': 0, 'neutral': 1, '-': 2, 'contradiction': 3}


def read_data():
    snli = read_snli_data()
    multinli = read_multinli()
    return snli, multinli


def read_snli_data():
    data = []
    for ds_name in ['train', 'dev', 'test']:
        LHS_words = torch.zeros(snli[ds_name]['length'], LHS_max_sent_len)
        RHS_words = torch.zeros(snli[ds_name]['length'], RHS_max_sent_len)
        labels = torch.zeros(snli[ds_name]['length'])
        sent_lens = torch.zeros(snli[ds_name]['length'])
        with open(snli[ds_name]['fname'], 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = json.loads(line)
                print(line)
    return


def read_multinli():
    return


if __name__ == '__main__':
    read_data()

# {'annotator_labels': ['entailment'],
#  'captionID': '3626642428.jpg#2',
#  'gold_label': 'entailment',
#  'pairID': '3626642428.jpg#2r1e',
#  'sentence1': 'there are six dogs in the water',
#  'sentence1_binary_parse': '( there ( are ( ( six dogs ) ( in ( the water ) ) ) ) )',
#  'sentence1_parse': '(ROOT (S (NP (EX there)) (VP (VBP are) (NP (NP (CD six) (NNS dogs)) (PP (IN in) (NP (DT the) (NN water)))))))',
#  'sentence2': 'There are animal wading in the waters.',
#  'sentence2_binary_parse': '( There ( ( are ( animal ( wading ( in ( the waters ) ) ) ) ) . ) )',
#  'sentence2_parse': '(ROOT (S (NP (EX There)) (VP (VBP are) (ADJP (JJ animal) (S (VP (VBG wading) (PP (IN in) (NP (DT the) (NNS waters))))))) (. .)))'}
