import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import code
import math
import sys
import numpy as np



class WordAveragingModel(nn.Module):
	def __init__(self, args):
		super(WordAveragingModel, self).__init__()
		self.embed = nn.Embedding(args.vocab_size, args.embedding_size)
		self.p_vector = nn.Parameter(torch.ones(args.embedding_size))
		self.embed.weight.data.uniform_(-1, 1)

	def forward(self, d, mask_d, att_dict=None, check_att=False):
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

		self.check_att = args.check_att
		self.att_type = args.att_type
		self.embed = nn.Embedding(args.vocab_size, args.embedding_size)


		if self.att_type.lower() in ["vanilla", "nearby"]:
			self.p_vector = nn.Parameter(torch.ones(args.embedding_size))
			self.w = nn.Parameter(torch.ones(args.embedding_size))
		elif self.att_type.lower() == "pos":
			self.p_vector = nn.Parameter(torch.ones(args.embedding_size + 1))
			self.w = nn.Parameter(torch.ones(args.embedding_size + 1))
		elif self.att_type.lower() == "pos_tag":
			self.pos_embed = nn.Embedding(args.pos_tag_vocab_size, args.pos_embedding_size)
			self.pos_embed.weight.data.uniform_(-1, 1)
			self.p_vector = nn.Parameter(torch.ones(args.embedding_size + args.pos_embedding_size))
			self.w = nn.Parameter(torch.ones(args.embedding_size + args.pos_embedding_size))
		elif self.att_type.lower() == "sentence_length":
			self.p_vector = nn.Parameter(torch.ones(args.embedding_size + 1))
			self.w = nn.Parameter(torch.ones(args.embedding_size))


		self.w.data.uniform_(-1,1)
		self.p_vector.data.uniform_(-1,1)
		self.embed.weight.data.uniform_(-1, 1)


	def forward(self, d, mask_d, pos=None, att_dict={}, check_att=False):
		d_embedded = self.embed(d)

		if self.att_type.lower() == "vanilla":
			w = self.w.unsqueeze(0).unsqueeze(1).expand_as(d_embedded)
			w = torch.sum(d_embedded * w, 2).squeeze(2) # B * T
		elif self.att_type.lower() == "pos":
			pos = torch.range(1, d.size(1))
			pos_sum = torch.sum(pos)
			# print(pos)
			pos = Variable((pos / pos_sum).unsqueeze(0).expand_as(d).unsqueeze(2))
			# print(pos.size())
			# print(d_embedded.size())
			d_embedded = torch.cat([d_embedded, pos], 2)
			w = self.w.unsqueeze(0).unsqueeze(1).expand_as(d_embedded)
			w = torch.sum(d_embedded * w, 2).squeeze(2) # B * T
		elif self.att_type.lower() == "pos_tag":
			pos_embedded = self.pos_embed(pos)
			print(d_embedded.size())
			print(pos_embedded.size())
			d_embedded = torch.cat([d_embedded, pos_embedded], 2)
			w = self.w.unsqueeze(0).unsqueeze(1).expand_as(d_embedded)
			w = torch.sum(d_embedded * w, 2).squeeze(2) # B * T

		elif self.att_type.lower() == "nearby":
			B, T, D = d_embedded.size()
			if T == 1:
				d_embedded = d_embedded * 0.8
			else:
				# print(d_embedded.size())
				prev_d_embedded = Variable(torch.zeros(B, T, D))
				after_d_embedded = Variable(torch.zeros(B, T, D))
				# print(d_embedded)
				a = d_embedded.data[:,:-1,:]
				prev_d_embedded.data[:,1:,:] = a
				after_d_embedded.data[:,:-1,:] = d_embedded.data[:,1:,:]
				d_embedded = 0.1 * prev_d_embedded + 0.1 * after_d_embedded + 0.8 * d_embedded
			w = self.w.unsqueeze(0).unsqueeze(1).expand_as(d_embedded)
			w = torch.sum(d_embedded * w, 2).squeeze(2) # B * T

		elif self.att_type.lower() == "sentence_length":
			w = self.w.unsqueeze(0).unsqueeze(1).expand_as(d_embedded)
			w = torch.sum(d_embedded * w, 2).squeeze(2) # B * T


		##### Softmax 1
		# code.interact(local=locals())
		w[mask_d == 0] = -999
		w = torch.exp(F.log_softmax(w)) * mask_d

		######## Softmax 2
		# w_max = torch.max(w, 1)[0].expand_as(w)
		# # w_max.data[w_max.data < 0] = 0
		# # code.interact(local=locals())
		# w2 = torch.exp(w - w_max)
		# if math.isnan(w.data[0,0]):
		# 	f = open("log.txt", "w")
		# 	f.write("\n#" * 50 + "self.w\n")
		# 	f.write(str(self.w))
		# 	f.write("\n#" * 50 + "d_embedded\n")
		# 	f.write(str(d_embedded))
		# 	f.write("\n#" * 50 + "w\n")
		# 	f.write(str(w))
		# 	f.write("\n#" * 50 + "w_max\n")
		# 	f.write(str(w_max))

		# 	f.write("\n#" * 50 + "params\n")
		# 	for param in self.parameters():
		# 		f.write(str(param.grad.data))
		# 	f.close()
		# 	exit(-1)
		# w = w2

		if check_att:
			# print("Checking att...")
			B, T = w.size()
			for i in range(B):
				for j in range(T):
					if mask_d.data[i,j] != 0:
						value = d.data[i,j]
						if not value in att_dict:
							att_dict[value] = []
						# print(value)
						att_dict[value].append(w.data[i,j])

		# code.interact(local=locals())

		# w_sum = torch.sum(w * mask_d, 1)
		# w_sum = w_sum.expand_as(w) 
		# w = (w / w_sum * mask_d)
		######## End of Softmax
		w = w.unsqueeze(2).expand_as(d_embedded)

		w_avg = torch.sum(d_embedded * w, 1).squeeze(1)
		if self.att_type.lower() == "sentence_length":
			d_length = torch.sum(mask_d, 1)
			w_avg = torch.cat([w_avg, d_length], 1)

		p_vec = self.p_vector.unsqueeze(0).expand_as(w_avg)
		w_avg = torch.sum(w_avg * p_vec, 1)
		if check_att:
			return F.sigmoid(w_avg), att_dict
		else:
			return F.sigmoid(w_avg)
		




