import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import pandas as pd
import tflearn
import tensorflow as tf
import random
import json
import pickle

from tensorflow.python.compiler.tensorrt import trt_convert as trt

with open("intents.json") as file:
    data = json.load(file)

try: 
    with open("intents.json") as file:
        data = json.load(file)

except:

    # Extracting Data

    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
        
        if intent["tag"] not in labels:
            labels.append(intent["tag"])


    # words stemming

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    # creating a bag of words

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
            
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    

# Use numpy to ocnvert our training data nad output to numpy arrays

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
            pickle.dump((words, labels, training, output), f)

# Now that we have preprocessed all of our data we are ready to start creating and training a model. 
# For our purposes we will use a fairly standard feed-forward neural network with two hidden layers. 
# The goal of our network will be to look at a bag of words and give a class that they belong too (one of our tags from the JSON file).

# We will start by defining the architecture of our model. 
# Keep in mind that you can mess with some of the numbers here and try to make an even better model! 
# A lot of machine learning is trial an error.

# tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# Here we fit the model. If the script can find an existing model, it will use that instead
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

# Making Predictions

# This makes the process to generate a response look like the following:
# – Get some input from the user
# – Convert it to a bag of words
# – Get a prediction from the model
# – Find the most probable class
# – Pick a response from that class

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(words.lower()) for words in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        
        results = model.predict([bag_of_words(inp, words)])
        results_index = np.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg["tag"] == tag:
                responses = tg["responses"]
        print(random.choice(responses))

chat()