import sys, os
import time
import utils
import config
import logging
import code
import numpy as np
from models.WordAveragingModel import *
from torch.autograd import Variable
import torch
from torch import optim
from torch.nn import MSELoss
import progressbar
import pickle

def eval(model, data, args):
	total_dev_batches = len(data)
	correct_count = 0.
	bar = progressbar.ProgressBar(max_value=total_dev_batches).start()
	loss = 0.

	print("total dev %d" % total_dev_batches)

	for idx, (mb_d, mb_mask_d, mb_l) in enumerate(data):
		# print(mb_d)
		mb_d = Variable(torch.from_numpy(mb_d)).long()
		# print("after")
		# print(mb_d)
		mb_mask_d = Variable(torch.from_numpy(mb_mask_d))
		mb_out = model(mb_d, mb_mask_d)
		# print(mb_out)

		batch_size = mb_d.size(0)
		mb_a = Variable(torch.Tensor(mb_l).type_as(mb_out.data)).view(batch_size, -1)
		loss += (- mb_a * torch.log(mb_out + 1e-9) - (1. - mb_a) * torch.log(1. - mb_out + 1e-9)).sum().data[0]
		# (torch.abs(mb_a - mb_out) * torch.log(torch.abs(mb_a - mb_out) + 1e-9)).sum().data[0]
		
		res = torch.abs(mb_a - mb_out) < 0.5
		# code.interact(local=locals())
		correct_count += res.sum().data[0]
		# print(correct_count)

		bar.update(idx+1)

	bar.finish()
	
	return correct_count, loss

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

	word_dict, args.vocab_size = utils.build_dict(train_examples[0], max_words=args.vocab_size)
	pickle.dump(word_dict, open(args.dict_file, "wb"))

	docs, labels = utils.encode(train_examples, word_dict)
	all_train = utils.gen_examples(docs, labels, args.batch_size)

	d_docs, d_labels = utils.encode(dev_examples, word_dict)
	total_dev = len(d_docs)
	all_dev = utils.gen_examples(d_docs, d_labels, args.batch_size)

	if args.test_only:
		test_examples = utils.load_data(args.test_file)
		args.num_test = len(test_examples[0])
		t_docs, t_labels = utils.encode(test_examples, word_dict)
		all_test = utils.gen_examples(t_docs, t_labels, args.batch_size)
	
	if os.path.exists(args.model_file):
		model = torch.load(args.model_file)
	elif args.model == "WordAveragingModel":
		model = WordAveragingModel(args)
	elif args.model == "WeightedWordAveragingModel":
		model = WeightedWordAveragingModel(args)

	if args.test_only:
		print("start evaluating on test")
		correct_count, loss = eval(model, all_test, args)
		print("test accuracy %f" % (float(correct_count) / float(args.num_test)))
		loss = loss / args.num_test
		print("test loss %f" % loss)

		correct_count, loss = eval(model, all_dev, args)
		print("dev accuracy %f" % (float(correct_count) / float(args.num_dev)))
		loss = loss / args.num_dev
		print("dev loss %f" % loss)
		return 0

	correct_count, loss = eval(model, all_dev, args)
	print("dev accuracy %f" % (float(correct_count) / float(args.num_dev)))
	loss = loss / args.num_dev
	print("dev loss %f" % loss)

	optimizer = optim.Adam(model.parameters(), lr=0.01)
	best_loss = loss
	for epoch in range(args.num_epoches):
		
		np.random.shuffle(all_train)

		bar = progressbar.ProgressBar(max_value=len(all_train), redirect_stdout=True)
		total_train_loss = 0.
		for idx, (mb_d, mb_mask_d, mb_l) in enumerate(all_train):

			mb_d = Variable(torch.from_numpy(mb_d)).long()
			mb_mask_d = Variable(torch.from_numpy(mb_mask_d))
			mb_out = model(mb_d, mb_mask_d)
			# print(mb_out)

			batch_size = mb_d.size(0)
			mb_a = Variable(torch.Tensor(mb_l).type_as(mb_out.data)).view(batch_size, -1)
			# cross entropy loss
			loss = (- mb_a * torch.log(mb_out + 1e-9) - (1. - mb_a) * torch.log(1. - mb_out + 1e-9)).sum() # / batch_size
			total_train_loss += loss.data[0]
			loss = loss / batch_size
			# loss = (torch.abs(mb_a - mb_out) * torch.log(torch.abs(mb_a - mb_out) + 1e-9)).sum() / batch_size
		
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			bar.update(idx+1)
		bar.finish()
		print("training loss: %f" % (total_train_loss / args.num_train))

		print("start evaluating on dev")
		# print(all_dev)
		correct_count, loss = eval(model, all_dev, args)
		print("correct count %f" % correct_count)
		print("total count %d" % args.num_dev)
		print("dev accuracy %f" % (float(correct_count) / float(args.num_dev)))
		loss = loss / args.num_dev
		print("dev loss %f" % loss)

		if loss < best_loss:
			torch.save(model, args.model_file)


if __name__ == "__main__":
	args = config.get_args()
	main(args)
