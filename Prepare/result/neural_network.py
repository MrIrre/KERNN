# -*- coding: utf-8 -*-
import torch


class MyNN(torch.nn.Module):
    def __init__(self, vocab_size, layers_num=6, embedding_size=64):
        super(MyNN, self).__init__()
        self.embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)  # padding_idx = 0 ???

        layers = []
        for i in range(layers_num):
            layers.append(torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, dilation=2**i, padding=2**i),
                torch.nn.Dropout(0.5),
                torch.nn.ReLU()))

        self.layers = torch.nn.ModuleList(layers)
        self.final_conv = torch.nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        # InputSize = BatchSize x MaxChunkLen | example: [32, 200]

        # batch_size, max_chunk_size = input.shape
        # print(f"Размер батча = {batch_size}")
        # print(f"Размер наибольшего текста = {max_text_size}")
        # print(f"Длина чанка максимальной длины = {max_chunk_size}")

        # print(input)

        x = self.embeddings(input)  # BatchSize x MaxChunkLen x EmbedSize | example: [32, 200, 64]
        # print(x)
        x = x.permute(0, 2, 1)  # BatchSize x EmbedSize x MaxInLen | example: [32, 64, 200]

        for i in range(len(self.layers)):
            layer = self.layers[i]
            x = x + layer(x)
            #print(f"После {i+1} слоя размерность тензора - {x.shape}")

        x = self.final_conv(x)  # BatchSize x 1 x MaxInLen | example: [32, 1, 200]
        # print(f"После финальной одномерной свертки размерность тензора - {x.shape}")

        x = x.squeeze(1)  # BatchSize x MaxInLen | example: [32, 200]
        # x = x.view(batch_size, max_chunk_size)  # BatchSize x MaxInLen | example: [32, 200]

        x = self.sigmoid(x)
        # print(x)
        # print(f"В конце размерность тензора - {x.shape}")

        return x

