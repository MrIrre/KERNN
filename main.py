import numpy as np
import pandas
import torch
import wikipediaapi
from pymystem3 import Mystem
import pymorphy2


class MyNN(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size=64):
        super(MyNN, self).__init__()
        self.embeds = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, padding_idx=0)

        self.conv1 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2**1)
        self.act1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(p=0.5)

        self.conv2 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2**2)
        self.act2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(p=0.5)

        self.conv3 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2**3)
        self.act3 = torch.nn.ReLU()
        self.dropout3 = torch.nn.Dropout(p=0.5)

        self.conv4 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2**4)
        self.act4 = torch.nn.ReLU()
        self.dropout4 = torch.nn.Dropout(p=0.5)

        self.conv5 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2**5)
        self.act5 = torch.nn.ReLU()
        self.dropout5 = torch.nn.Dropout(p=0.5)

        self.conv6 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2**6)
        self.act6 = torch.nn.ReLU()
        self.dropout6 = torch.nn.Dropout(p=0.5)

        # TODO: Add Conv with kernel_size = 1

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embeds(x)
        x = self.dropout1(self.act1(self.conv1(x)))
        x = self.dropout2(self.act2(self.conv2(x)))
        x = self.dropout3(self.act3(self.conv3(x)))
        x = self.dropout4(self.act4(self.conv4(x)))
        x = self.dropout5(self.act5(self.conv5(x)))
        x = self.dropout6(self.act6(self.conv6(x)))

        # TODO: Add Conv with kernel_size = 1

        x = self.softmax(x)

        return x


def main():
    pass


if __name__ == '__main__':
    main()
