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

def eval(model, data, args):
	total_dev_batches = len(data)
	correct_count = 0.
	bar = progressbar.ProgressBar(max_value=total_dev_batches).start()
	total_loss = 0.

	print("total dev %d" % total_dev_batches)

	for idx, (mb_d, mb_mask_d, mb_l) in enumerate(data):
		print(idx+1)
		mb_d = Variable(torch.from_numpy(mb_d)).long()
		mb_mask_d = Variable(torch.from_numpy(mb_mask_d))
		mb_out = model(mb_d, mb_mask_d)

		batch_size = mb_d.size(0)
		mb_a = Variable(torch.Tensor(mb_l).type_as(mb_out.data)).view(batch_size, -1)
		# print(mb_a.size(), mb_out.size())
		total_loss += torch.log(torch.abs(mb_a - mb_out)).sum().data[0]
		

		res = torch.abs(mb_a - mb_out) < 0.5
		correct_count += res.sum().data[0]

		bar.update(idx+1)

	bar.finish()
	print("accuracy %f" % (float(correct_count) / float(args.num_dev)))
	loss = total_loss / args.num_dev

	return loss	

def main(args):
	
	# train_examples contains (docs, questions, candidates, answers)
	if args.debug:
		train_examples = utils.load_data(args.train_file, max_example=args.max_train)
		dev_examples = utils.load_data(args.dev_file, max_example=args.max_dev)
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
	
	if os.path.exists(args.model_file):
		model = torch.load(args.model_file)
	else:
		model = WordAveragingModel(args)

	print("start evaluating on dev")
	dev_loss = eval(model, all_dev, args)

	optimizer = optim.Adam(model.parameters(), lr=0.01)
	best_loss = 999999
	for epoch in range(args.num_epoches):
		np.random.shuffle(all_train)

		bar = progressbar.ProgressBar(max_value=len(all_train), redirect_stdout=True)
		for idx, (mb_d, mb_mask_d, mb_l) in enumerate(all_train):

			mb_d = Variable(torch.from_numpy(mb_d)).long()
			mb_mask_d = Variable(torch.from_numpy(mb_mask_d))
			mb_out = model(mb_d, mb_mask_d)

			batch_size = mb_d.size(0)
			mb_a = Variable(torch.Tensor(mb_l).type_as(mb_out.data)).view(batch_size, -1)
			# print(mb_a.size(), mb_out.size())
			loss = torch.log(torch.abs(mb_a - mb_out).sum() / batch_size)
		
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			bar.update(idx+1)

		print("start evaluating on dev")
		dev_loss = eval(model, all_dev, args)

		if dev_loss < best_loss:
			torch.save(model, args.model_file)


if __name__ == "__main__":
	args = config.get_args()
	main(args)
