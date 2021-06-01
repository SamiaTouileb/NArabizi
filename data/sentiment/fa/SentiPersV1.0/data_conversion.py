import csv
import numpy as np

from lxml import etree
from lxml.etree import fromstring
import os
import json
import re

from nltk.tokenize import TreebankWordTokenizer as twt

parser = etree.XMLParser(recover=True, encoding='utf8')


def extract_data(xml_file):
    data = []
    mark_xml = open(xml_file).read().encode('utf8')
    base_root = fromstring(mark_xml, parser)
    review = base_root.find("Review")
    for sent in review:
        try:
            label = int(sent.get("Value"))
            text = sent.text
            data.append((label, text))
        except ValueError:
            pass
    return data


# Get the texts and labels
data = []

for base in os.listdir("SentiPersV1.0/Data"):
    xml_file = os.path.join("SentiPersV1.0/Data", base)
    data.extend(extract_data(xml_file))


# remove neutral and mixed and convert labels to 1 (pos) or 0 (neg)
positive = []
negative = []
for label, text in data:
    if label > 1:
        positive.append([1, text])
    if label < 0:
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

