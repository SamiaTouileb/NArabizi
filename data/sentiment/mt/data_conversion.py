import csv
from nltk import FreqDist
import numpy as np

import xlrd

def majority_label(labels):
    fd = FreqDist(labels)
    if len(fd) == 1:
        return labels[0]
    else:
        (label1, label1_count), (label2, label2_count) = fd.most_common(2)
        if label1_count > label2_count:
            return label1
        else:
            return "mixed"

workbook = xlrd.open_workbook("microblog_dataset_all_annotations.xlsx")
sheet = workbook.sheet_by_index(0)

texts = []
labels = []

# Get the texts and majority labels
for rowx in range(sheet.nrows):
    values = sheet.row_values(rowx)
    texts.append(values[0])
    labels.append(majority_label(values[1:]))

# remove neutral and mixed and convert labels to 1 (pos) or 0 (neg)
positive = []
negative = []
for label, text in zip(labels, texts):
    if label == "positive":
        positive.append([1, text])
    if label == "negative":
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

