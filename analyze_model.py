import sys, os
import time
import utils
import config
import logging
import code
import numpy as np
from models.WordAveragingModel import WordAveragingModel
from torch.autograd import Variable
import torch
from torch import optim
from torch.nn import MSELoss
import progressbar
import pickle

model = torch.load("model.th")

data = model.embed.weight.data

print(data.size())



# print(norm)




# print(model.embed.weight.data)

word2id = pickle.load(open( "dict.pkl", "rb" ))
id2word = {word2id[key]: key for key in word2id}
print(len(id2word))

print("words with max norms")
norm = torch.norm(data, 2, 1)
for i in range(10):
	max_index = norm.max(0)[1][0,0]
	print(id2word[norm.max(0)[1][0,0]])
	norm[max_index] = 0


norm = torch.norm(data, 2, 1)
print("words with min norms")
for i in range(10):
	min_index = norm.min(0)[1][0,0]
	print(id2word[norm.min(0)[1][0,0]])
	norm[min_index] = 99999
# print(id2word)

# code.interact(local=locals())

# print(dict)