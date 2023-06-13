import sys
#import pandas as pd
import re
#import numpy as np
import os
#from collections import Counter
import logging
import time
import pickle
from random import sample
from keras.utils import pad_sequences
# m = open('tokenizer.pkl', 'rb')
# print(m.read())
# model = pickle.load(m)
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
TRAIN_SIZE = 0.8

# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

# WORD2VEC 
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10

# KERAS
SEQUENCE_LENGTH = 300
EPOCHS = 1
BATCH_SIZE = 1024

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

# EXPORT
KERAS_MODEL = "model.h5"
WORD2VEC_MODEL = "model.w2v"
TOKENIZER_MODEL = "tokenizer.pkl"
ENCODER_MODEL = "encoder.pkl"

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
from keras.models import load_model
model = load_model('model.h5')



def decode_sentiment(score, include_neutral=True):
      if include_neutral:        
          label = NEUTRAL
          if score <= SENTIMENT_THRESHOLDS[0]:
              label = NEGATIVE
          elif score >= SENTIMENT_THRESHOLDS[1]:
              label = POSITIVE
  
          return label
      else:
          return NEGATIVE if score < 0.5 else POSITIVE
def predict(text, include_neutral=True):
    start_at = time.time()
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)

    return {"label": label, "score": float(score),
        "elapsed_time": time.time()-start_at}
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from pprint import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import praw
import asyncio
user_agent = "Scraper 1.0 by /u/Crimson_Raiderz"
reddit = praw.Reddit(
client_id = "pNgJ18VKAZsJrD1MYPULtg",
client_secret = "bJCCS_ACh7VWwUorn27eWn7H_6K0QQ",
user_agent=user_agent
)

titleComment = dict()
comments = set()
f = open('t.pkl', 'rb')
saved = pickle.load(f)
titleComment = saved['t']
# .top(time_filter="month")
sub = reddit.subreddit("StockMarket").top(time_filter="month")
for submission in sub:
    #print(str(submission.title.encode('utf-8')))
    for s in submission.comments:
        print(s)
        comments.add(str(s.body.encode('utf-8')))
    titleComment[str(submission.title.encode('utf-8'))] = comments
posit = []
negat = []
print('hi')
count = 0
t = sample(list(titleComment),int(len(titleComment)/40))
for x in titleComment:
    posCount = 0
    negCount = 0
    t2 = sample(titleComment[x], int(len(titleComment[x])/40))
    for y in t2:
        print(len(t), count, len(t2))
        count += 1
        if predict(y)['label'] == 'POSITIVE': posCount += 1
        else: negCount += 1
    count = 0
    if posCount > negCount: posit.append(x)
    else: negat.append(x)
print('hi2')
lines = []
for p in posit: lines.append("Positive: " + p)
for n in negat: lines.append("Negative: " + n)
with open('stocklines.txt', 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')