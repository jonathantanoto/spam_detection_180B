import os
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D,\
 Dense, Dropout, LSTM, Bidirectional

# supress warnings
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# load in model and tokenizer
print("---Building Model---")
reconstructed_model = tf.keras.models.load_model("../models/model")
with open('../models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

print('Model Building success.')

print()

print("---SPAM PREDICTOR---")

while True:
    print()
    print()
    x = input("Type a message to see probability of being a spam: ")

    input_lst = [x]

    def predict(input_lst):
        seq = tokenizer.texts_to_sequences(input_lst)
        padded = pad_sequences(seq, maxlen = 50,
                        padding = 'post',
                        truncating = 'post')
        return reconstructed_model.predict(padded)

    print()

    res = 100*predict(input_lst)[0][0]

    verdict = lambda x: 'Spam' if x>50 else 'Not Spam'

    print("Verdict: " + verdict(res))

    print("Probability of being spam: " + str(res))
    print()

    print("Ctrl + C to exit")
    print("-------------")
