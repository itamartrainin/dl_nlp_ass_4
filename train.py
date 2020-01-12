from model import SkipConnBiLSTM
import preprocess

def train(args):
    return


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-p', '--preprocess', action='store_true')
    parser.add_argument('--embds-fname', help="Path to parsed embeddings file.", default="../data/embds")
    parser.add_argument('--data-fname', help="Path to parsed data file.", default="../data/preprocessed_data")
    parser.add_argument('-m', '--model-fname', help="Path to model file.", default="")
    args = parser.parse_args()

    embds, data, word_to_ix, label_to_ix = preprocess.get_preprocessed_data(args.embds_fname, args.data_fname, args.preprocess, args.preprocess)
    snli = data['snli']
    multinli = data['multinli']

    print(args)

    # train(args)
