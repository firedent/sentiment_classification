import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as topti
import string
from pathlib import Path
from torchtext import data
from torchtext.vocab import GloVe
from imdb_dataloader import IMDB


# Class for creating the neural network.
class Network(tnn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.lstm = tnn.LSTM(50, 100, 2, batch_first=True, dropout=0.5)
        tnn.init.orthogonal_(self.lstm.all_weights[0][0])
        tnn.init.orthogonal_(self.lstm.all_weights[0][1])
        tnn.init.orthogonal_(self.lstm.all_weights[1][0])
        tnn.init.orthogonal_(self.lstm.all_weights[1][1])
        self.fc1 = tnn.Linear(100, 64)
        tnn.init.orthogonal_(self.fc1.weight)
        self.fc2 = tnn.Linear(64, 1)
        tnn.init.orthogonal_(self.fc2.weight)
        #
        # super(Network, self).__init__()
        # self.maxpool = torch.nn.MaxPool1d(4)
        # self.relu = torch.nn.ReLU()
        # self.conv1 = tnn.Conv1d(in_channels=50, out_channels=50, kernel_size=8, padding=5)
        # self.conv2 = tnn.Conv1d(in_channels=50, out_channels=50, kernel_size=8, padding=5)
        # self.conv3 = tnn.Conv1d(in_channels=50, out_channels=50, kernel_size=8, padding=5)
        # self.maxpool_global = torch.nn.AdaptiveMaxPool1d(1)
        # self.fc1 = tnn.Linear(50, 1)

    def forward(self, input, length):
        o = input
        o = self.lstm(o)[0]
        o = o[range(len(length)), length - 1]
        o = self.fc1(o)
        o = F.relu(o)
        o = self.fc2(o)
        return o.squeeze()
        #
        # o = input.transpose(1, 2)
        # o = self.maxpool(self.relu(self.conv1(o)))
        # o = self.maxpool(self.relu(self.conv2(o)))
        # o = self.maxpool_global(torch.relu(self.conv3(o)))
        # o = self.fc1(o.squeeze())
        # return o.squeeze()

def lossFunc():
    """
    Define a loss function appropriate for the above networks that will
    add a sigmoid to the output and calculate the binary cross-entropy.
    """
    return tnn.BCEWithLogitsLoss()


def measures(outputs, labels):
    """
    TODO:
    Return (in the following order): the number of true positive
    classifications, true negatives, false positives and false
    negatives from the given batch outputs and provided labels.
    outputs and labels are torch tensors.
    """
    tp, tn, fp, fn = (0,) * 4
    for o, l in zip(outputs, labels):
        if o == 0:
            if torch.randint(0, 2, (1,)) == 0:
                o = 1
            else:
                o = -1
        if o < 0:
            if l == 1:
                fn += 1
            else:
                tn += 1
        elif o > 0:
            if l == 1:
                tp += 1
            else:
                fp += 1

    return tp, tn, fp, fn


def evaluation(epoch, net, device, dev, testLoader, textField):
    net.train(False)
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
    # computations and reduce memory usage.
    with torch.no_grad():
        for batch in testLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            outputs = net(inputs, length)

            tp_batch, tn_batch, fp_batch, fn_batch = measures(outputs, labels)
            true_pos += tp_batch
            true_neg += tn_batch
            false_pos += fp_batch
            false_neg += fn_batch

    accuracy = 100 * (true_pos + true_neg) / len(dev)
    matthews = MCC(true_pos, true_neg, false_pos, false_neg)
    net.train(True)
    print("Classification accuracy: %.2f%%\n"
          "Matthews Correlation Coefficient: %.2f" % (accuracy, matthews))

    logFilePath = Path('./training.log')
    with open(logFilePath, mode='a+') as f:
        print("Epoch%d:\n"
              "Classification accuracy: %.2f%%\n"
              "Matthews Correlation Coefficient: %.2f\n" % (epoch+1, accuracy, matthews),
              file=f)
    torch.save(net.state_dict(), f"./model_epoch{epoch+1}.pth")
    print("Saved model")


class PreProcessing():
    def pre(x):
        printbale = set([i for i in string.ascii_letters + string.digits])
        return [''.join(c for c in t if c in printbale) for t in x]

    def post(batch, vocab):
        """Called after numericalization but prior to vectorization"""
        return batch

    text_field = data.Field(lower=True, include_lengths=True, batch_first=True, preprocessing=pre, postprocessing=post)


def main():

    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Load the training dataset, and create a data loader to generate a batch.
    textField = PreProcessing.text_field
    labelField = data.Field(sequential=False)

    train, dev = IMDB.splits(textField, labelField, train="train", validation="dev")

    textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev)

    trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=16,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)

    net = Network().to(device)

    criterion =lossFunc()
    optimiser = topti.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)  # Minimise the loss using the Adam algorithm.

    for epoch in range(30):
        running_loss = 0

        for i, batch in enumerate(trainLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device),\
                                     batch.text[1].to(device),\
                                     batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # PyTorch calculates gradients by accumulating contributions to them (useful for
            # RNNs).  Hence we must manually set them to zero before calculating them.
            optimiser.zero_grad()

            # Forward pass through the network.
            output = net(inputs, length)

            loss = criterion(output, labels)

            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            running_loss += loss.item()

            if i % 128 == 127:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f" % (epoch + 1, i + 1, running_loss / 32))
                running_loss = 0

        evaluation(epoch, net, device, dev, testLoader, textField)

    # Save mode
    print("Saved model")
    torch.save(net.state_dict(), "./model.pth")


# Matthews Correlation Coefficient calculation.
def MCC(tp, tn, fp, fn):
    numerator = tp * tn - fp * fn
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(numerator, denominator)


if __name__ == '__main__':
    main()
