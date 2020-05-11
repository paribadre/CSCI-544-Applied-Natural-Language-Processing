
import glob
import re
import os
from collections import Counter
import collections
from math import log


# class NaiveBayes:

# def __init__(self,root)
all_files = glob.glob(os.path.join('op_spam_training_data', '*/*/*/*.txt'))
CountDeceptive = Counter()
CountTruthful = Counter()
CountPositive = Counter()
CountNegative = Counter()
ProbDeceptive = Counter()
ProbTruthful = Counter()
ProbPositive = Counter()
ProbNegative = Counter()
Vocabulary = []

StopWords = ["ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out", "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", "itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", "below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", "then", "that", "because", "what", "over", "why", "so", "can", "did", "not", "now", "under", "he", "you", "herself", "has", "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "t", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than","would", ""]

test_by_class = collections.defaultdict(list)
train_by_class = collections.defaultdict(list)

words = []

# for f in all_files:
#   # Take only last 4 components of the path. The earlier components are useless
#   # as they contain path to the classes directories.
#   class1, class2, fold, fname = f.split('/')[-4:]
#   if fold == 'fold1':
#     # True-clause will not enter in Vocareum as fold1 wont exist, but useful for your own code.
#     test_by_class[class1+class2].append(f)
#   else:
#     train_by_class[class1+class2].append(f)
# #

Deceptive = []
Truthful = []
Positive = []
Negative = []
Test = []

for i in all_files:
    if "fold1" not in i:
        if "deceptive_from_MTurk" in i:
            Deceptive.append(i)
        else:
            Truthful.append(i)
        if "negative_polarity" in i:
            Negative.append(i)
        else:
            Positive.append(i)


for i in Deceptive:
    with open(i, 'r') as ff:
      text = ff.read()
    text = re.sub('[^A-Za-z]+', ' ', text)
    text = text.lower()
    words = text.split(" ")
    for j in words:
        if j not in StopWords:
            CountDeceptive[j] += 1
            if j not in Vocabulary:
                Vocabulary.append(j)
            words.remove(j)
# print(CountDeceptive)

for i in Truthful:
    with open(i, 'r') as ff:
      text = ff.read()
    text = re.sub('[^A-Za-z]+', ' ', text)
    text = text.lower()
    words = text.split(" ")
    for j in words:
        if j not in StopWords:
            CountTruthful[j] += 1
            if j not in Vocabulary:
                Vocabulary.append(j)
            words.remove(j)

for i in Positive:
    with open(i, 'r') as ff:
      text = ff.read()
    text = re.sub('[^A-Za-z]+', ' ', text)
    text = text.lower()
    words = text.split(" ")
    for j in words:
        if j not in StopWords:
            CountPositive[j] += 1
            words.remove(j)

for i in Negative:
    with open(i, 'r') as ff:
      text = ff.read()
    text = re.sub('[^A-Za-z]+', ' ', text)
    text = text.lower()
    words = text.split(" ")
    for j in words:
        if j not in StopWords:
            CountNegative[j] += 1
            words.remove(j)


PriorDeceptive = len(Deceptive)/(len(Deceptive) + len(Truthful))
PriorTruthful = len(Truthful)/(len(Deceptive) + len(Truthful))
PriorPositive = len(Positive)/(len(Positive) + len(Negative))
PriorNegative = len(Negative)/(len(Positive) + len(Negative))
VocabCount = len(Vocabulary)

for i in Vocabulary:
    ProbDeceptive[i] = (CountDeceptive[i] + 1)/ (sum(CountDeceptive.values()) + VocabCount +1)

for i in Vocabulary:
    ProbTruthful[i] = (CountTruthful[i] + 1)/ (sum(CountTruthful.values()) + VocabCount + 1)

for i in Vocabulary:
    ProbPositive[i] = (CountPositive[i] + 1)/ (sum(CountPositive.values()) + VocabCount + 1)

for i in Vocabulary:
    ProbNegative[i] = (CountNegative[i] + 1)/ (sum(CountNegative.values()) + VocabCount + 1)

f = open("nbmodel.txt", "a")
f.write(str(ProbDeceptive))
f.write("\n")
f.write(str(ProbTruthful))
f.write("\n")
f.write(str(ProbPositive))
f.write("\n")
f.write(str(ProbNegative))

# print(PriorPositive, PriorDeceptive, PriorNegative, PriorTruthful)
# print(CountPositive)
# print(CountNegative)
# print(CountTruthful)
# print(CountDeceptive)
# print(ProbDeceptive)
# print(ProbTruthful)
# print(ProbPositive)
# print(ProbNegative)
# print(len(Vocabulary))
# print(test_by_class)
Prob1 = 1.0
Prob2 = 1.0
Prob3 = 1.0
Prob4 = 1.0

for i in all_files:
    if "fold1" in i:
        Test.append(i)

for i in Test:
    with open(i, 'r') as ff:
      text = ff.read()
    text = re.sub('[^A-Za-z]+', ' ', text)
    text = text.lower()
    words = text.split(" ")
    for j in words:
        if j not in StopWords:
             Prob1 += log(ProbPositive[j])
             Prob2 += log(ProbNegative[j])
             Prob3 += log(ProbTruthful[j])
             Prob4 += log(ProbDeceptive[j])
    Prob1 += log(PriorPositive)
    Prob2 += log(PriorNegative)
    Prob3 += log(PriorTruthful)
    Prob4 += log(PriorDeceptive)
    print(Prob1, Prob2, Prob3, Prob4)
    if Prob1 > Prob2:
        x = "Positive"
    else:
        x = "Negative"
    if Prob3 > Prob4:
        x = x + "Truthful"
    else:
        x = x + "Deceptive"

    print(x + "Path =" + i)




### Print the file names (i.e. the dictionaries of classes and their filenames)
# import json

# print('\n\n *** Test data:')
# print(json.dumps(test_by_class, indent=2))
# print('\n\n *** Train data:')
