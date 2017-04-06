import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import code
import math
import sys



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
		self.w.data.uniform_(-1,1)

		self.p_vector = nn.Parameter(torch.ones(args.embedding_size))
		self.p_vector.data.uniform_(-1,1)
		self.embed.weight.data.uniform_(-1, 1)

		self.check_att = args.check_att
		self.att_type = args.att_type

	def forward(self, d, mask_d, att_dict={}):
		d_embedded = self.embed(d)

		w = self.w.unsqueeze(0).unsqueeze(1).expand_as(d_embedded)
		w = torch.sum(d_embedded * w, 2).squeeze(2) # B * T
		w_max = torch.max(w, 1)[0].expand_as(w)
		# w_max.data[w_max.data < 0] = 0
		# code.interact(local=locals())
		w2 = torch.exp(w - w_max)
		if math.isnan(w.data[0,0]):
			f = open("log.txt", "w")
			f.write("\n#" * 50 + "self.w\n")
			f.write(str(self.w))
			f.write("\n#" * 50 + "d_embedded\n")
			f.write(str(d_embedded))
			f.write("\n#" * 50 + "w\n")
			f.write(str(w))
			f.write("\n#" * 50 + "w_max\n")
			f.write(str(w_max))

			f.write("\n#" * 50 + "params\n")
			for param in self.parameters():
				f.write(str(param.grad.data))
			f.close()
			exit(-1)
		w = w2

		if self.check_att:
			B, T = w.size()
			for i in range(B):
				for j in range(T):
					if mask_d.data[i,j] != 0:
						value = d.data[i,j]
						if not value in att_dict:
							att_dict[value] = []
						att_dict[value].append(w.data[i,j])

		w_sum = torch.sum(w * mask_d, 1)
		w_sum = w_sum.expand_as(w) 
		w = (w / w_sum * mask_d).unsqueeze(2).expand_as(d_embedded)
		
		w_avg = torch.sum(d_embedded * w, 1).squeeze(1)
		p_vec = self.p_vector.unsqueeze(0).expand_as(w_avg)
		w_avg = torch.sum(w_avg * p_vec, 1)
		return F.sigmoid(w_avg)
		




