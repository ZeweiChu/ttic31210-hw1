#!/bin/bash

python main.py --batch_size 128 --train_file /Users/zeweichu/UCHI/ttic31210/data/senti.binary.train.txt --dev_file /Users/zeweichu/UCHI/ttic31210/data/senti.binary.dev.txt --model_file wordavg.th --model WordAveragingModel --learning_rate 0.01 --eval_epoch 1 --optimizer Adam
