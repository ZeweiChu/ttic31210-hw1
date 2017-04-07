import logging
import numpy as np
from collections import Counter
import itertools


def load_data(in_file, max_example=None, relabeling=True):
    docs = []
    labels = []
    num_examples = 0
    f = open(in_file, 'r')
    line = f.readline()
    while line != "": 
        # print(line)
        line = line.strip().split("\t") 
        
        if len(line) >= 2:
            docs.append(line[0].split())
            labels.append(line[1])
            num_examples += 1
        else:
            docs.append(line[0].split())
            num_examples += 1

        if (max_example is not None) and (num_examples >= max_example):
            break
        line = f.readline()
    f.close()
    return (docs, labels)

def build_dict(sentences, max_words=50000):
    word_count = Counter()
    for sent in sentences:
        for w in sent:
            word_count[w] += 1
    ls = word_count.most_common(max_words)
    total_words = len(ls) + 1
    return {w[0]: index+1 for (index, w) in enumerate(ls)}, total_words

def encode(examples, word_dict, pos_examples=None, pos_dict=None, sort_by_len=True):
    '''
        Encode the sequences. 
    '''
    in_doc = []
    in_l = []
    in_pos = []

    
    if pos_examples is not None:
        for idx, (d_words, l_words, pos_words) in enumerate(zip(examples[0], examples[1], pos_examples[0])):
            seq1 = [word_dict[w] if w in word_dict else 0 for w in d_words]
            seq2 = [int(w) for w in l_words]
            seq3 = [pos_dict[w] if w in pos_dict else 0 for w in pos_words]
            
            if len(seq1) > 0:
                in_doc.append(seq1)
                in_l.append(seq2)
                in_pos.append(seq3)
    else:
        for idx, (d_words, l_words) in enumerate(zip(examples[0], examples[1])):
            seq1 = [word_dict[w] if w in word_dict else 0 for w in d_words]
            seq2 = [int(w) for w in l_words]
            
            if len(seq1) > 0:
                in_doc.append(seq1)
                in_l.append(seq2)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        # sort by the document length
        sorted_index = len_argsort(in_doc)
        in_doc = [in_doc[i] for i in sorted_index]
        in_l = [in_l[i] for i in sorted_index]
        if pos_examples is not None:
            in_pos = [in_pos[i] for i in sorted_index]

    if pos_examples is not None:
        return in_doc, in_l, in_pos
    else:
        return in_doc, in_l

def get_minibatches(n, minibatch_size, shuffle=False):
    idx_list = np.arange(0, n, minibatch_size)
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for idx in idx_list:
        minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
    return minibatches

def prepare_data(seqs):
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)
    max_len = np.max(lengths)
    x = np.zeros((n_samples, max_len)).astype('int32')
    x_mask = np.zeros((n_samples, max_len)).astype('float32')
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
        x_mask[idx, :lengths[idx]] = 1.0
    return x, x_mask

def gen_examples(d, l, batch_size, pos=None):

    minibatches = get_minibatches(len(d), batch_size)
    all_ex = []
    for minibatch in minibatches:
        mb_d = [d[t] for t in minibatch]
        mb_l = [l[t] for t in minibatch]
        mb_d, mb_mask_d = prepare_data(mb_d)
        if pos is not None:
            mb_pos = [pos[t] for t in minibatch]
            mb_pos, mb_mask_pos = prepare_data(mb_pos)
            all_ex.append((mb_d, mb_mask_d, mb_l, mb_pos))
        else:
            all_ex.append((mb_d, mb_mask_d, mb_l))
    return all_ex


