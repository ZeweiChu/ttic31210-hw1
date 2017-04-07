#!/bin/bash

python main.py --batch_size 128 --train_file /Users/zeweichu/UCHI/ttic31210/data/senti.binary.train.txt --dev_file /Users/zeweichu/UCHI/ttic31210/data/senti.binary.dev.txt --test_file /Users/zeweichu/UCHI/ttic31210/data/senti.binary.test.txt --test_only True --model WeightedWordAveragingModel --model_file WeightedWordAveragingModel.th --dict_file WeightedWordAveragingModelDict.pkl  
