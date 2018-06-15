import json
import os

import numpy as np
import json
from itertools import chain
import pandas as pd
from PIL import Image
import pickle

label1 = {}

filename = "labeled.txt"
with open(filename) as f:
    for line in f:
        item = line.strip().split("\t")
        label1["106_" + item[0]] = item[1]

label2 = {}
filename = "labeled2.txt"
with open(filename) as f:
    for line in f:
        item = line.strip().split("\t")
        label2["057_" + item[0]] = item[1]

json1 = json.dumps(label1)
json2 = json.dumps(label2)

with open("gesture_labeled.json","w") as f:
    json.dump(label1,f)
    json.dump(label2,f)