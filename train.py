from model import SkipConnBiLSTM
import preprocess
import torch
import torch.nn as nn
# data:
#     snli:
#         train: tensor
#         dev: tensor
#         test: tensor
#     multinli:
#         train: tensor
#         dev_matched: tensor
#         dev_mismatched: tensor
# tensor_dim -> (num_examples, len_seq)

def train(embds, data, word_to_ix, label_to_ix, **kwargs):

    # Read data into variables
    snli_train = data["snli"]["train"]
    snli_dev = data["snli"]["dev"]

    multinli_train = data["multinli"]["train"]
    multinli_dev_matched = data["multinli"]["dev_matched"]
    multinli_dev_mismatched = data["multinli"]["dev_mismatched"]

    batch_size = kwargs['batch_size']
    epochs = kwargs['epochs']
    ignore_index = label_to_ix[preprocess.pad_string]

    snli_sample_size = torch.floor(snli_train[0].size(0) * 0.15)
    num_of_examples = snli_sample_size + multinli_train[0].size(0)

    loss_function = nn.NLLLoss(ignore_index=ignore_index)

    snli_rand_idx = torch.randperm(snli_train[0].size(0))
    snli_train_shuffled = tuple(map(lambda x: x[snli_rand_idx], snli_train))

    multinli_rand_idx = torch.randperm(multinli_train[0].size(0))
    multinli_train_shuffled = tuple(map(lambda x: x[multinli_rand_idx], multinli_train))

    for epoch in range(epochs):

        snli_train_samples = tuple(map(lambda x: x[epoch * snli_train_samples:(epoch + 1) * snli_sample_size], snli_train_shuffled))
        unified_train_samples = tuple(map(lambda x: torch.cat(x, dim=0), zip(snli_train_samples, multinli_train_shuffled)))
        for batch in range(0, num_of_examples, batch_size):
            multinli_batch_samples = tuple(map(lambda x: x[batch:batch + batch_size], multinli_train))


    # Initialize linear layers
    # ignore_index = label_to_ix[preprocess.pad_string]
    # loss_function = nn.NLLLoss(ignore_index=ignore_index)

    # Needs to be in train and lr should be dynamic
    # lr = kwargs['lr']
    # optimizer = optim.Adam(self.parameters(), lr=self.lr) ???how to used adam???

    return


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--embds-fname', help="Path to parsed embeddings file.", default="../data/embds")
    parser.add_argument('--data-fname', help="Path to parsed data file.", default="../data/preprocessed_data")
    parser.add_argument('-m', '--model-fname', help="Path to model file.", default="../data/model")
    parser.add_argument('-p', '--preprocess', help="Should load preprocessed data.", action='store_true')
    parser.add_argument('-f', '--fine-tune', help="Should fine tune the embeddings.", action='store_false')
    parser.add_argument('-d', '--drop-out', help="Dropout between bilstm layers.", type=float, default=0.1)
    parser.add_argument('-b', '--batch-size', help="Batch Size.", type=float, default=32)
    parser.add_argument('-e', '--epochs', help="Number of epochs.", type=float, default=100)
    parser.add_argument('--lr', help="Learning Rate", type=float, default=0.0002)
    parser.add_argument('-i', '--ignore_index', help="Learning Rate", type=float, default=0.0002)
    parser.add_argument('-a', '--activation', help="Learning Rate", default='tanh')
    parser.add_argument('-s', '--shortcuts', help="Learning Rate", default='all')
    parser.add_argument('--h', help="BiLSTM hidden layers dimensions", nargs='+', type=int, default=[512])
    parser.add_argument('--lin-h', help="Linear hidden layers dimensions", nargs='+', type=int, default=[1])
    args = parser.parse_args()
    args = vars(args)   # Convert namespace to dictionary

    args['LHS_max_len'] = preprocess.LHS_max_sent_len
    args['RHS_max_len'] = preprocess.RHS_max_sent_len

    embds, data, word_to_ix, label_to_ix = preprocess.get_preprocessed_data(args['embds_fname'], args['data_fname'],
                                                                            args['preprocess'], args['preprocess'])

    train(embds, data, word_to_ix, label_to_ix, **args)

    print(args)
