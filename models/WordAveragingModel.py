import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import code

class WordAveragingModel(nn.Module):
	def __init__(self, args):
		super(WordAveragingModel, self).__init__()
		self.embed = nn.Embedding(args.vocab_size, args.embedding_size)
		self.p_vector = nn.Parameter(torch.ones(args.embedding_size))
		self.embed.weight.data.uniform_(-1, 1)

	def forward(self, d, mask_d):
		d_embedded = self.embed(d)
		mask_d = mask_d.unsqueeze(2).expand_as(d_embedded)
		d_embedded = d_embedded * mask_d

		d_sum = torch.sum(d_embedded, 1)
		d_length = torch.sum(mask_d, 1)
		avg = (d_sum / d_length).squeeze(1)
		
		p_vec = self.p_vector.unsqueeze(0).expand_as(avg)
		# print(avg.size(), p_vec.size())
		avg = torch.sum(avg * p_vec, 1)
		# code.interact(local=locals())
		# print(avg)
		return F.sigmoid(avg)

class WeightedWordAveragingModel(nn.Module):
	def __init__(self, args):
		super(WeightedWordAveragingModel, self).__init__()
		self.embed = nn.Embedding(args.vocab_size, args.embedding_size)

		self.w = nn.Parameter(torch.ones(args.embedding_size))

		self.p_vector = nn.Parameter(torch.ones(args.embedding_size))
		self.embed.weight.data.uniform_(-1, 1)

	def forward(self, d, mask_d):
		d_embedded = self.embed(d)
		# mask_d3 = mask_d.unsqueeze(2).expand_as(d_embedded)
		# d_embedded = d_embedded * mask_d3


		w = self.w.unsqueeze(0).unsqueeze(1).expand_as(d_embedded)
		w = torch.sum(d_embedded * w, 2).squeeze(2) # B * T
		# print('#' * 50)
		# print(w.size())
		# print(torch.max(w, 1))
		w_max = torch.max(w, 1)[0]
		# w_max = w_max.unsqueeze(0)

		# print(w.size(), w_max.size())
		w_max = w_max.expand_as(w)
		w_max.data[w_max.data < 0] = 0
		# code.interact(local=locals())
		w = torch.exp(w - w_max)
		# print(w)
		w_sum = torch.sum(w * mask_d, 1)
		w_sum = w_sum.expand_as(w) 
		w = w / w_sum * mask_d
		w = w.unsqueeze(2).expand_as(d_embedded)

		# print(w.size())
		w_avg = torch.sum(d_embedded * w, 1).squeeze(1)
		# print(w_avg.size())

		# code.interact(local=locals())
		
		p_vec = self.p_vector.unsqueeze(0).expand_as(w_avg)
		# print(p_vec.size())
		# print(avg.size(), p_vec.size())
		w_avg = torch.sum(w_avg * p_vec, 1)
		# code.interact(local=locals())
		# print(avg)
		return F.sigmoid(w_avg)
		




