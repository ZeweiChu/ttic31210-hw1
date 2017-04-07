#!/bin/bash

python main.py --check_att True --batch_size 128 --train_file /Users/zeweichu/UCHI/ttic31210/data/senti.binary.train.txt --dev_file /Users/zeweichu/UCHI/ttic31210/data/senti.binary.dev.txt --max_train 10000 --max_dev 1000 --model WeightedWordAveragingModel --model_file WeightedWordAveragingModel.th --dict_file WeightedWordAveragingModelDict.pkl --learning_rate 0.005 --optimizer Adam
