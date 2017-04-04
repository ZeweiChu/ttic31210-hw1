import sys
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


def main(args):
	
	# train_examples contains (docs, questions, candidates, answers)
	if args.debug:
		train_examples = utils.load_data(args.train_file, max_example=10000)
		dev_examples = utils.load_data(args.dev_file, max_example=10000)
	else: 
		train_examples = utils.load_data(args.train_file)
		dev_examples = utils.load_data(args.dev_file)

	args.num_train = len(train_examples[0])
	args.num_dev = len(dev_examples[0])

	word_dict = utils.build_dict(train_examples[0], max_words=args.vocab_size)
	docs, labels = utils.encode(train_examples, word_dict)
	all_train = utils.gen_examples(docs, labels, args.batch_size)


	d_docs, d_labels = utils.encode(dev_examples, word_dict)
	total_dev = len(d_docs)
	all_dev = utils.gen_examples(d_docs, d_labels, args.batch_size)
	

	model = WordAveragingModel(args)

	

	optimizer = optim.Adam(model.parameters(), lr=0.01)

	for epoch in range(args.num_epoches):
		np.random.shuffle(all_train)

		bar = progressbar.ProgressBar(max_value=len(docs)/args.batch_size)
		for idx, (mb_d, mb_mask_d, mb_l) in enumerate(all_train):

			mb_d = Variable(torch.from_numpy(mb_d)).long()
			mb_mask_d = Variable(torch.from_numpy(mb_mask_d))
			mb_out = model(mb_d, mb_mask_d)

			batch_size = mb_d.size(0)
			mb_a = Variable(torch.Tensor(mb_l).type_as(mb_out.data)).view(batch_size, -1)
			# print(mb_a.size(), mb_out.size())
			loss = -torch.log(torch.abs(mb_a - mb_out).sum() / batch_size)
		
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			bar.update(idx)


		correct_count = 0.

		for idx, (mb_d, mb_mask_d, mb_l) in enumerate(all_dev):
			mb_d = Variable(torch.from_numpy(mb_d)).long()
			mb_mask_d = Variable(torch.from_numpy(mb_mask_d))
			mb_out = model(mb_d, mb_mask_d)
			batch_size = mb_d.size(0)
			mb_a = Variable(torch.Tensor(mb_l).type_as(mb_out.data)).view(batch_size, -1)
			res = torch.abs(mb_a - mb_out)
			# print(mb_a)
			# print(mb_out)
			print(res)
			res = res < 0.5
			print(res)
			
			# res[res >= 0.5] = 0
			
			correct_count += res.sum().data[0]
		

		print("dev accuracy %f" % (float(correct_count) / float(total_dev)))


def eval():
	pass

if __name__ == "__main__":
	args = config.get_args()
	main(args)
