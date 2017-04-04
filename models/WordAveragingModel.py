import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class WordAveragingModel(nn.Module):
	def __init__(self, args):
		super(WordAveragingModel, self).__init__()

		self.embed = nn.Embedding(args.vocab_size, args.embedding_size)
		self.p_vector = nn.Parameter(torch.zeros(args.embedding_size))


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
		return F.sigmoid(avg)


		




