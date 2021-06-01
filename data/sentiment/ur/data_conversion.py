import csv
import numpy as np

import os
import json
import re

data = []

for line in open("urdu-sentiment-corpus-v1.tsv"):
    text, label = line.strip()[:-1].strip(), line.strip()[-1]
    data.append([label, text])

# remove neutral and mixed and convert labels to 1 (pos) or 0 (neg)
positive = []
negative = []
for label, text in data[1:]:
    if label == "P":
        positive.append([1, text])
    if label == "N":
        negative.append([0, text])

# create train dev test splits
pos_train_idx = int(len(positive) * .7)
pos_dev_idx = int(len(positive) * .8)
neg_train_idx = int(len(negative) * .7)
neg_dev_idx = int(len(negative) * .8)

pos_train = positive[:pos_train_idx]
pos_dev = positive[pos_train_idx:pos_dev_idx]
pos_test = positive[pos_dev_idx:]

neg_train = negative[:neg_train_idx]
neg_dev = negative[neg_train_idx:neg_dev_idx]
neg_test = negative[neg_dev_idx:]

train = pos_train + neg_train
dev = pos_dev + neg_dev
test = pos_test + neg_test

np.random.shuffle(train)
np.random.shuffle(dev)
np.random.shuffle(test)

# print to csv
with open("train.csv", "w") as outfile:
    writer = csv.writer(outfile)
    for line in train:
        writer.writerow(line)

with open("dev.csv", "w") as outfile:
    writer = csv.writer(outfile)
    for line in dev:
        writer.writerow(line)

with open("test.csv", "w") as outfile:
    writer = csv.writer(outfile)
    for line in test:
        writer.writerow(line)

