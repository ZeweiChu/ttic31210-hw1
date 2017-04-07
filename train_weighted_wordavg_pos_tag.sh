#!/bin/bash

python main_pos_tag.py --batch_size 128 --train_file /Users/zeweichu/UCHI/ttic31210/data/senti.binary.train.txt --dev_file /Users/zeweichu/UCHI/ttic31210/data/senti.binary.dev.txt --train_pos_file /Users/zeweichu/UCHI/ttic31210/data/senti.binary.train.tag --dev_pos_file /Users/zeweichu/UCHI/ttic31210/data/senti.binary.dev.tag --max_train 10000 --max_dev 1000 --model WeightedWordAveragingModel --model_file WeightedWordAveragingModelPosTag.th --dict_file WeightedWordAveragingModelDict.pkl --learning_rate 0.01 --optimizer Adam --eval_epoch 1 --att_type pos_tag
