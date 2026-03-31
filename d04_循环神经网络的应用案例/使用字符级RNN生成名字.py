from __future__ import unicode_literals, print_function, division
from io import open
import glob  # path 读取指定路径下的特有格式的文件
import os
import unicodedata
import string

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # Plus EOS marker


# 统一 成 英文
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


print(unicodeToAscii("O'Néàl"))


def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')  # strip 去掉前后空格
    return [unicodeToAscii(line) for line in lines]  #语法糖


# → index
category_lines = {}
all_categories = []


# script_dir = os.path.dirname(os.path.abspath(__name__))
# print(script_dir)
# print(os.getcwd())
def findFiles(path: os.path) -> list:
    return glob.glob(path)


# for filename in findFiles('data/names/*.txt'):
#     print( filename)

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)
print('# categories:', n_categories, all_categories)

import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        # i2h input → hidden语义 语言+单词+语义
        # i2o input → output输出 output_size = input_size
        # o2o output  单词+语义 单词
        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)  # 过拟合
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


import random


# 列表中的随机项
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]  # eg 0-10 7


# 从该类别中获取随机类别和随机行
def randomTrainingPair():
    #语言
    category = randomChoice(all_categories)
    # 语言下的单词
    line = randomChoice(category_lines[category])
    return category, line


# 类别的One-hot张量
def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor


# 用于输入的从头到尾字母（不包括EOS）的one-hot矩阵
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)  # 1 batch 一个个输进去所以默认batch为1
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor


# 用于目标的第二个结束字母（EOS）的LongTensor
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)


# Alt A l  l t  t <EOS>

# 从随机(类别，行)对中创建类别，输入和目标张量
def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor


criterion = nn.NLLLoss()

learning_rate = 0.0005


def train(category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()

    rnn.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item() / input_line_tensor.size(0)


import time
import math


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# 模型保存路径
MODEL_PATH = 'rnn_name_generator.pth'
USE_SAVED_MODEL = True  # 设置为 True 直接加载已训练好的模型，False 则重新训练


# rnn = RNN(n_letters, 128, n_letters)
#
# n_iters = 10000
# print_every = 500
# plot_every = 200
# all_losses = []
# total_loss = 0  # Reset every plot_every iters
#
# start = time.time()
#
# for iter in range(1, n_iters + 1):
#     output, loss = train(*randomTrainingExample())
#     total_loss += loss
#
#     if iter % print_every == 0:
#         print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))
#
#     if iter % plot_every == 0:
#         all_losses.append(total_loss / plot_every)
#         total_loss = 0

rnn = RNN(n_letters, 128, n_letters)

n_iters = 10000
print_every = 500
plot_every = 200
all_losses = []
total_loss = 0  # Reset every plot_every iters

# 检查是否需要训练
if USE_SAVED_MODEL and os.path.exists(MODEL_PATH):
    print(f'正在加载已训练好的模型：{MODEL_PATH}')
    checkpoint = torch.load(MODEL_PATH)
    rnn.load_state_dict(checkpoint['model_state_dict'])
    all_losses = checkpoint['losses']
    print('模型加载完成！')
else:
    print('开始训练模型...')
    start = time.time()

    for iter in range(1, n_iters + 1):
        output, loss = train(*randomTrainingExample())
        total_loss += loss

        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0

    print('训练完成！正在保存模型...')

    # 保存模型参数
    torch.save({
        'model_state_dict': rnn.state_dict(),
        'losses': all_losses,
        'n_iters': n_iters,
    }, MODEL_PATH)

    print(f'模型已保存到：{MODEL_PATH}')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib

matplotlib.use('TkAgg')

plt.figure()
plt.plot(all_losses)
plt.show()

max_length = 20


# 来自类别和首字母的样本
def sample(category, start_letter='A'):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = categoryTensor(category)  # 获取类别张量
        input = inputTensor(start_letter)
        hidden = rnn.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:  #<EOS>
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        return output_name


# 从一个类别和多个起始字母中获取多个样本
def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))


samples('Russian', 'RUS')

samples('German', 'GER')

samples('Spanish', 'SPABDEFGHIK')

samples('Chinese', 'CHI')
