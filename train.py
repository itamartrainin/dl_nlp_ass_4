from model import SkipConnBiLSTM
import preprocess

def train(embds, data, word_to_ix, label_to_ix, **kwargs):
    print(kwargs)

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
