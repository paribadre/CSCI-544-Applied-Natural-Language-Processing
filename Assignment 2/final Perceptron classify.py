import glob
import re
import os
from collections import Counter
import collections
import ast
import sys
dics = []
cnt = Counter()
with open(sys.argv[1], 'r') as model:
    for line in model:
        dics.append(line)
    bias1 = ast.literal_eval(dics[0])
    WeightPN = ast.literal_eval(dics[1])
    bias2 = ast.literal_eval(dics[2])
    WeightTD = ast.literal_eval(dics[3])

all_files = glob.glob(os.path.join(sys.argv[2], '*/*/*/*.txt'))

for i in all_files:
    ActivationPN = bias1
    ActivationTD = bias2
    cnt = Counter()
    with open(i, 'r') as ff:
        text = ff.read()
    text = re.sub('[^A-Za-z]+', ' ', text)
    text = text.lower()
    words = text.split(" ")
    cnt.update(words)
    for key, value in cnt.items():
        if key in WeightPN:
            ActivationPN += WeightPN[key]*value
            ActivationTD += WeightTD[key]*value
    if ActivationTD > 0:
        x = "truthful "
    else:
        x = "deceptive "
    if ActivationPN > 0:
        x += "positive "
    else:
        x += "negative "
    f = open("percepoutput.txt", "a+")
    f.write(x + i)
    f.write("\n")
    f.close()
