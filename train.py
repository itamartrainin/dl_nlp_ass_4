import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt

from model import SkipConnBiLSTM
import preprocess

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

def accuracy(model, loss_func, data):

    s1, s2, lens1, lens2, labels = data

    log_probs = model.forward(s1, s2, lens1, lens2)

    loss = loss_func(log_probs, labels)
    prediciton = torch.argmax(log_probs, dim=1)
    acc = torch.sum(prediciton == labels).float() / float(len(labels))
    return loss.item(), acc * 100

def train(model, embds, data, word_to_ix, label_to_ix, device, **kwargs):

    # Read data into variables
    snli_train = data["snli"]["train"]

    multinli_train = data["multinli"]["train"]
    multinli_dev_matched = data["multinli"]["dev_matched"]
    multinli_dev_mismatched = data["multinli"]["dev_mismatched"]

    # Upload all data to device
    # snli_train = tuple(map(lambda x: x.to(device), snli_train))

    # multinli_train = tuple(map(lambda x: x.to(device), multinli_train))
    # multinli_dev_matched = tuple(map(lambda x: x.to(device), multinli_dev_matched))
    # multinli_dev_mismatched = tuple(map(lambda x: x.to(device), multinli_dev_mismatched))

    batch_size = kwargs['batch_size']
    epochs = kwargs['epochs']
    ignore_index = label_to_ix[preprocess.pad_string]
    lr = kwargs['lr']
    lr_delta = kwargs['lr_delta']
    epoch_drop_it = kwargs['epoch_drop_it']

    snli_sample_size = math.floor(snli_train[0].size(0) * 0.15)
    num_of_examples = snli_sample_size + multinli_train[0].size(0)

    loss_function = nn.NLLLoss(ignore_index=ignore_index)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    snli_rand_idx = torch.randperm(snli_train[0].size(0))
    snli_train_shuffled = tuple(map(lambda x: x[snli_rand_idx], snli_train))

    multinli_rand_idx = torch.randperm(multinli_train[0].size(0))
    multinli_train_shuffled = tuple(map(lambda x: x[multinli_rand_idx], multinli_train))

    train_loss_arr, train_acc_arr, dev_matched_loss_arr, dev_matched_acc_arr, dev_mismatched_loss_arr, dev_mismatched_acc_arr = [], [], [], [], [], []

    for epoch in range(epochs):

        snli_train_samples = tuple(map(lambda x: x[epoch * snli_sample_size:(epoch + 1) * snli_sample_size], snli_train_shuffled))
        unified_train_samples = tuple(map(lambda x: torch.cat(x, dim=0), zip(snli_train_samples, multinli_train_shuffled)))
        avg_train_loss_batches, avg_train_acc_batches = 0, 0
        num_of_examples = unified_train_samples[0].shape[0]
        for batch in range(0, unified_train_samples[0].shape[0], batch_size):
            if batch < num_of_examples - batch_size:
                s1, s2, lens1, lens2, labels = tuple(map(lambda x: x[batch:batch + batch_size].to(device), unified_train_samples))
            else:
                s1, s2, lens1, lens2, labels = tuple(map(lambda x: x[batch:].to(device), unified_train_samples))
            log_probs = model.forward(s1, s2, lens1, lens2)

            loss = loss_function(log_probs, labels)
            loss.backward()
            optimizer.step()

            avg_train_loss_batches += loss.item() * (len(labels) / num_of_examples)
            prediciton = torch.argmax(log_probs, dim=1)
            avg_train_acc_batches += (torch.sum(prediciton == labels).float() / float(len(labels))) * (len(labels) / num_of_examples) * 100

        train_loss_arr.append(avg_train_loss_batches)
        train_acc_arr.append(avg_train_acc_batches)
        # If (epoch % epoch_drop_it) is 0 cut the lerning rate by lr_delta otherwise keep it the same.
        lr = (epoch % epoch_drop_it == 0) * lr * lr_delta + (epoch % epoch_drop_it != 0) * lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with torch.no_grad():

            dev_matched_loss, dev_matched_acc = accuracy(model, loss_function, multinli_dev_matched)
            dev_mismatched_loss, dev_mismatched_acc = accuracy(model, loss_function, multinli_dev_mismatched)
            dev_matched_loss_arr.append(dev_matched_loss)
            dev_matched_acc_arr.append(dev_matched_acc)
            dev_mismatched_loss_arr.append(dev_mismatched_loss)
            dev_mismatched_acc_arr.append(dev_mismatched_acc)
            print("Epoch: {} |"
                  " Dev Matched Loss: {} Dev Matched Acc: {} |"
                  " Dev Mismatched Loss: {} Dev Mismatched Acc: {}"
                  .format(epoch, dev_matched_loss, dev_matched_acc, dev_mismatched_loss, dev_mismatched_acc))

    learning_info = {"train loss": train_loss_arr,
                       "train acc": train_acc_arr,
                       "dev matched loss": dev_matched_loss_arr,
                       "dev matched acc": dev_matched_acc_arr,
                       "dev mismatched loss": dev_mismatched_loss_arr,
                       "dev mismatched acc": dev_mismatched_acc_arr}

    return model, learning_info

def learning_plots(learning_info):
    for key in learning_info.keys():
        plt.figure()
        plt.plot(learning_info[key])
        plt.title(key)
        plt.xlabel("Epochs")
        plt.savefig("{}.jpeg".format(key))


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--embds-fname', help="Path to parsed embeddings file.", default="../data/embds")
    parser.add_argument('--data-fname', help="Path to parsed data file.", default="../data/preprocessed_data")
    parser.add_argument('-m', '--model-fname', help="Path to model file.", default="../data/model")
    parser.add_argument('-p', '--preprocess', help="Should load preprocessed data.", action='store_true')
    parser.add_argument('-f', '--fine-tune', help="Should fine tune the embeddings.", action='store_false')
    parser.add_argument('-d', '--dropout', help="Dropout between bilstm layers.", type=float, default=0.1)
    parser.add_argument('-b', '--batch-size', help="Batch Size.", type=float, default=32)
    parser.add_argument('-e', '--epochs', help="Number of epochs.", type=float, default=100)
    parser.add_argument('--epoch_drop_it', help="Number of epochs until lr decay.", type=int, default=2)
    parser.add_argument('--lr', help="Learning Rate.", type=float, default=0.0002)
    parser.add_argument('--lr-delta', help="Change rate in learning rate.", type=float, default=0.5)
    parser.add_argument('-i', '--ignore_index', help="Label of '-'.", type=int, default=0)
    parser.add_argument('-a', '--activation', help="Activation type ('tanh'/'relu').", default='relu')
    parser.add_argument('-s', '--shortcuts', help="Shortcuts state ('all'/'word'/'none').", default='all')
    parser.add_argument('--h', help="BiLSTM hidden layers dimensions.", nargs='+', type=int, default=[350])
    parser.add_argument('--lin-h', help="Linear hidden layers dimensions.", nargs='+', type=int, default=[400])
    args = parser.parse_args()
    args = vars(args)   # Convert namespace to dictionary

    args['LHS_max_len'] = preprocess.LHS_max_sent_len
    args['RHS_max_len'] = preprocess.RHS_max_sent_len

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embds, data, word_to_ix, label_to_ix = preprocess.get_preprocessed_data(args['embds_fname'], args['data_fname'],
                                                                            args['preprocess'], args['preprocess'])
    # embds, data, word_to_ix, label_to_ix = preprocess.get_preprocessed_data(args['embds_fname'], args['data_fname'],
    #                                                                         args['preprocess'], False)

    model = SkipConnBiLSTM(embds, (word_to_ix, label_to_ix), args['h'], args['lin_h'], args).to(device)

    trained_model, learning_info = train(model, embds, data, word_to_ix, label_to_ix, device, **args)
    learning_plots(learning_info)
    torch.save([trained_model, learning_info], 'model_experiment')

    print('DONE')