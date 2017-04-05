#!/bin/bash

python main.py --debug True --batch_size 32 --train_file /Users/zeweichu/UCHI/ttic31210/data/senti.binary.train.txt --dev_file /Users/zeweichu/UCHI/ttic31210/data/senti.binary.dev.txt --max_train 3000 --max_dev 1000 --model WeightedWordAveragingModel --model_file WeightedWordAveragingModel.th
