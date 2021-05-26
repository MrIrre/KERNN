from typing import Any

import torch
import math


class MyNN(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size=64):
        super(MyNN, self).__init__()
        self.embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)  # padding_idx = 0 ???

        self.conv1 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, dilation=2**0, padding=2**0)
        self.act1 = torch.nn.ReLU()
        # self.norm1 = torch.nn.LayerNorm(max_in_length)
        self.dropout1 = torch.nn.Dropout(p=0.3)

        self.conv2 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, dilation=2**1, padding=2**1)
        self.act2 = torch.nn.ReLU()
        # self.norm2 = torch.nn.LayerNorm(max_in_length)
        self.dropout2 = torch.nn.Dropout(p=0.3)

        self.conv3 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, dilation=2**2, padding=2**2)
        self.act3 = torch.nn.ReLU()
        # self.norm3 = torch.nn.LayerNorm(max_in_length)
        self.dropout3 = torch.nn.Dropout(p=0.3)

        self.conv4 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, dilation=2**3, padding=2**3)
        self.act4 = torch.nn.ReLU()
        # self.norm4 = torch.nn.LayerNorm(max_in_length)
        self.dropout4 = torch.nn.Dropout(p=0.3)

        self.conv5 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, dilation=2**4, padding=2**4)
        self.act5 = torch.nn.ReLU()
        # self.norm5 = torch.nn.LayerNorm(max_in_length)
        self.dropout5 = torch.nn.Dropout(p=0.3)

        self.conv6 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, dilation=2**5, padding=2**5)
        self.act6 = torch.nn.ReLU()
        # self.norm6 = torch.nn.LayerNorm(max_in_length)
        self.dropout6 = torch.nn.Dropout(p=0.3)

        self.final_conv = torch.nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1)  # TODO: CHECK
        # self.final_dropout = torch.nn.Dropout(p=0.5)
        # self.final_conv2 = torch.nn.Conv1d(in_channels=256, out_channels=1, kernel_size=1)

        # self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        # InputSize = BatchSize x MaxTextLen x MaxChunkLen | example: [32, 20, 100]

        batch_size, max_text_size, max_chunk_size = input.shape
        # print(f"Размер батча = {batch_size}")
        # print(f"Размер наибольшего текста = {max_text_size}")
        # print(f"Длина чанка максимальной длины = {max_chunk_size}")

        # print(input)
        x = input.view(batch_size * max_text_size, max_chunk_size)  # BatchSize*MaxTextLen x MaxChunkLen | example: [32*20, 100]

        x = self.embeddings(x)  # BatchSize*MaxTextLen x MaxChunkLen x EmbedSize | example: [32*20, 200, 64]
        # print(x)
        x = x.permute(0, 2, 1)  # BatchSize*MaxTextLen x EmbedSize x MaxInLen | example: [32*20, 64, 200]

        #print(f"Перед слоями размерность тензора - {x.shape}")
        x = self.conv1(x)
        # print(self.conv1.weight)
        #print(f"После первой свёртки - {x.shape}")

        x = self.act1(x)
        x = self.dropout1(x)  # TODO: CHECK
        # x = self.norm1(x)

        #print(f"После 1 слоя размерность тензора - {x.shape}")
        x = self.conv2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        # x = self.norm2(x)

        #print(f"После 2 слоя размерность тензора - {x.shape}")

        x = self.act3(self.conv3(x))
        x = self.dropout3(x)
        # x = self.norm3(x)
        #
        # #print(f"После 3 слоя размерность тензора - {x.shape}")
        #
        x = self.act4(self.conv4(x))
        x = self.dropout4(x)
        # x = self.norm4(x)
        #
        # #print(f"После 4 слоя размерность тензора - {x.shape}")
        #
        x = self.act5(self.conv5(x))
        x = self.dropout5(x)
        # x = self.norm5(x)
        #
        # #print(f"После 5 слоя размерность тензора - {x.shape}")
        #
        x = self.act6(self.conv6(x))
        x = self.dropout6(x)
        # x = self.norm6(x)

        #print(f"После 6 слоя размерность тензора - {x.shape}")

        x = self.final_conv(x)  # BatchSize*MaxTextLen x 1 x MaxInLen | example: [32*20, 1, 200]

        # print('----')

        # x = self.final_dropout(x)  # TODO: CHECK
        # x = self.final_conv2(x)

        # print(f"После финальной одномерной свертки размерность тензора - {x.shape}")

        x = x.squeeze(1)  # BatchSize*MaxTextLen x MaxInLen | example: [32*20, 200]
        x = x.view(batch_size, max_text_size, max_chunk_size)  # BatchSize x MaxTextLen x MaxInLen | example: [32, 20, 200]

        x = self.sigmoid(x)
        # print(x)

        # print(f"В конце размерность тензора - {x.shape}")

        # x = threshold_func(x, 0.3)

        # res.append(x)

        res = x
        # res = torch.stack(res)

        return res

    @staticmethod
    def get_save_padding(in_channels, out_channels, kernel_size, stride):
        res = math.ceil((out_channels * stride - (in_channels - kernel_size + stride)) / 2)
        # print(res)
        return res


# class ThresholdFunc(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, tensor, threshold):
#         return (tensor >= threshold).float()
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_input = grad_output.clone()
#         # print('Custom backward called!')
#         return grad_input, None
#
#
# threshold_func = ThresholdFunc.apply
