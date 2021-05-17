# import pymystem3
#
# mystem = pymystem3.Mystem()
#
# text = "Иван Иванович Иванов приехал в Санкт-Петербург"
# for word in text.split(" "):
#     with mystem.analyze(word) as result:
#         print(result[0].form,
#              "({0})".format(result[0]),
#              result[0].stem_grammemes,
#              result[0].flex_grammemes
#         )

import torch

DELIM = "------------------------"

x = torch.tensor([
    [[1., 2., 3.],
     [4., 5., 6.]],
    [[4., 5., 6.],
     [7., 8., 9.]],
    [[1., 2., 3.],
     [7., 8., 9.]],
    [[2., 3., 1.],
     [7., 8., 9.]]
])

print(f"X shape is {x.shape}")
print(DELIM)

conv = torch.nn.Conv1d(2, 3, kernel_size=3, padding=1)
conv.weight = torch.nn.Parameter(torch.tensor(
    [
        [[1., 0., 1.],
         [0., 1., 0.]],
        [[2., 0., 2.],
         [0., 2., 0.]],
        [[3., 0., 3.],
         [0., 3., 0.]]
    ]
))
conv.bias = torch.nn.Parameter(torch.tensor(
    [0.1, 0.2, 0.3]
))

print("Conv1D weights is")
print(conv.weight)
print()
print("Conv1D bias is")
print(conv.bias)
print(DELIM)

res = conv(x)
print("Result is")
print(res)


# [[[ 6.1000,  9.1000,  8.1000],
#   [12.2000, 18.2000, 16.2000],
#   [18.3000, 27.3000, 24.3000]],
#
#  [[12.1000, 18.1000, 14.1000],
#   [24.2000, 36.2000, 28.2000],
#   [36.3000, 54.3000, 42.3000]],
#
#  [[ 9.1000, 12.1000, 11.1000],
#   [18.2000, 24.2000, 22.2000],
#   [27.3000, 36.3000, 33.3000]],
#
#  [[10.1000, 11.1000, 12.1000],
#   [20.2000, 22.2000, 24.2000],
#   [30.3000, 33.3000, 36.3000]]]
