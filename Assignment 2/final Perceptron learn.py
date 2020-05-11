
import glob
import re
import os
from collections import Counter
import collections
import sys
from random import shuffle
import random

all_files = glob.glob(os.path.join(sys.argv[1] , '*/*/*/*.txt'))
Vocabulary = Counter()
Vocab = Counter()
FeatureCount = 0
WeightPN = dict()
WeightTD = dict()
WeightAvgPN = dict()
WeightAvgTD = dict()
UpdatePN = dict()
UpdateTD = dict()
Beta1 = 0
Beta2 = 0
b = 0
Token = 1
bias1 = 0
bias2 = 0
bias3 = 0
bias4 = 0
MaxIter = 55
train_by_class = collections.defaultdict(list)
words = []
c = 0

for f in all_files:
    class1, class2, fold, fname = f.split('/')[-4:]
    if "negative_polarity" in class1:
        c1 = -1
    else:
        c1 = 1
    if "deceptive_from_MTurk" in class2:
        c2 = -1
    else:
        c2 = 1
    train_by_class[c].append(c1)
    train_by_class[c].append(c2)
    train_by_class[c].append(Counter())
    with open(f, 'r') as ff:
        text = ff.read()
    text = re.sub('[^A-Za-z]+', ' ', text)
    text = text.lower()
    words = text.split(" ")
    train_by_class[c][2].update(words)
    Vocabulary.update(words)
    c += 1

for i in Vocabulary:
    if 7 < Vocabulary[i] < 1500:
        Vocab[i] = Vocabulary[i]

for key, value in Vocab.items():
    WeightPN[key] = 0
    WeightTD[key] = 0
    WeightAvgPN[key] = 0
    WeightAvgTD[key] = 0
    UpdateTD[key] = 0
    UpdatePN[key] = 0
    FeatureCount += 1

for i in range(MaxIter):
    random.Random(MaxIter*7).shuffle(train_by_class)
    for j in train_by_class:
        ActivationPN = 0
        ActivationTD = 0
        for key, value in train_by_class[j][2].items():
            if key in WeightPN:
                ActivationPN += WeightPN[key]*value
                ActivationTD += WeightTD[key]*value
        ActivationPN += bias1
        ActivationTD += bias2
        if train_by_class[j][0]*ActivationPN <= 0:
            for key, value in WeightPN.items():
                WeightPN[key] += train_by_class[j][0]*train_by_class[j][2][key]
                UpdatePN[key] += train_by_class[j][0] * train_by_class[j][2][key] * Token
            bias1 += train_by_class[j][0]
            Beta1 += train_by_class[j][0] * Token
        if train_by_class[j][1]*ActivationTD <= 0:
            for key, value in WeightTD.items():
                WeightTD[key] += train_by_class[j][1]*train_by_class[j][2][key]
                UpdateTD[key] += train_by_class[j][1] * train_by_class[j][2][key] * Token
            bias2 += train_by_class[j][1]
            Beta2 += train_by_class[j][0]*Token
        Token += 1

for key, value in WeightAvgPN.items():
    WeightAvgPN[key] = WeightPN[key] - (UpdatePN[key]/Token)

for key, value in WeightAvgTD.items():
    WeightAvgTD[key] = WeightTD[key] - (UpdateTD[key]/Token)

bias3 = bias1 - Beta1/Token
bias4 = bias2 - Beta2/Token

f1 = open("vanillamodel.txt", "w")
f1.write(str(bias1))
f1.write("\n")
f1.write(str(WeightPN))
f1.write("\n")
f1.write(str(bias2))
f1.write("\n")
f1.write(str(WeightTD))
f1.write("\n")
f1.close()
f2 = open("averagedmodel.txt", "w")
f2.write(str(bias3))
f2.write("\n")
f2.write(str(WeightAvgPN))
f2.write("\n")
f2.write(str(bias4))
f2.write("\n")
f2.write(str(WeightAvgTD))
f2.write("\n")
f2.close()

