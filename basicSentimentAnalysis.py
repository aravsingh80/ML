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
    if label == "NEUTRAL": label = "NEGATIVE"

    return {"label": label, "score": float(score),
        "elapsed_time": time.time()-start_at}
print("Life is good: "+predict("Life is good")['label'])
print("Life is bad: "+predict("Life is bad")['label'])
print("Life exists: "+predict("Life exists")['label'])
print("The stock did terrible: "+predict("The stock plunged")['label'])