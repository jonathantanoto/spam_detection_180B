import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout, LSTM, Bidirectional

import pickle

# supress warnings
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


def build():
    # load data
    print("---Loading Data---")
    df = pd.read_csv('data/enron_spam_data.zip', compression='zip',\
    header=0, sep=',', quotechar='"')
    df = df.drop('Message ID', axis=1)

    print('Shape of Data: ' + str(df.shape))
    print(df['Spam/Ham'].value_counts())
    print()

    # binarizing labels
    print("---Processing Data---")
    df['Spam/Ham']= df['Spam/Ham'].map({'ham': 0, 'spam': 1})
    label = df['Spam/Ham'].values

    # Split data train/test
    train_msg, test_msg, train_labels, test_labels =\
    train_test_split(df['Message'], label, test_size=0.2, random_state=12)
    train_msg, test_msg = train_msg.astype(str), test_msg.astype(str)

    print()

    # Use Keras to tokenize, declare hyperparameters
    print("---Tokenizing---")
    tokenizer = Tokenizer(num_words = 1000, #500
        char_level=False, oov_token = "<OOV>")
    tokenizer.fit_on_texts(train_msg)

    word_index = tokenizer.word_index

    # check how many words 
    index_length = len(word_index)
    print('There are %s unique tokens in training data. ' % index_length)

    # saving
    if not os.path.exists("models"):
        os.mkdir("models")

    with open('models/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL) 

    print()

    # Sequencing and padding on both sets
    print("---Sequencing and Padding---")
    training_sequences = tokenizer.texts_to_sequences(train_msg)
    training_padded = pad_sequences(training_sequences, maxlen = 50,\
    padding = 'post', truncating = 'post' )
    testing_sequences = tokenizer.texts_to_sequences(test_msg)
    testing_padded = pad_sequences(testing_sequences, maxlen = 50,\
    padding = 'post', truncating = 'post')

    # Reshaping labels for LSTM 
    train_labels_mod = np.asarray(train_labels).astype('float32').reshape((-1,1))
    test_labels_mod = np.asarray(test_labels).astype('float32').reshape((-1,1))

    # Shape of train tensor
    print('Training tensor shape: ', training_padded.shape)
    print('Testing tensor shape: ', testing_padded.shape)

    print()

    # Bi-directional LSTM Spam detection architecture
    print("---Training Neural Network Model---")
    model = Sequential()
    model.add(Embedding(1000, 16, input_length=50))
    model.add(Bidirectional(LSTM(20, dropout=0.2,\
    return_sequences=True)))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(loss = 'binary_crossentropy', optimizer = 'adam',\
    metrics=['accuracy'])

    num_epochs = 30
    early_stop = EarlyStopping(monitor='val_loss', patience=2)
    history = model.fit(training_padded, train_labels_mod, epochs=num_epochs, 
                        validation_data=(testing_padded, test_labels_mod),
                        callbacks =[early_stop], verbose=2)

    print()

    # save model
    model.save("models/model")

    print("Model Saved.")