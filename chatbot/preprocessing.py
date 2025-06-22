import json
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from pathlib import Path

stemmer = PorterStemmer()

# Tokenize a sentence
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Stem a word
def stem(word):
    return stemmer.stem(word.lower())

# Convert a sentence into a Bag-of-Words vector
def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    return np.array([1 if w in tokenized_sentence else 0 for w in all_words], dtype=np.float32)

# Load and preprocess data
def load_training_data(json_path):
    with open(json_path, 'r') as f:
        intents = json.load(f)

    all_words = []
    tags = []
    xy = []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            tokens = tokenize(pattern)
            all_words.extend(tokens)
            xy.append((tokens, tag))

    # Stem and filter punctuation
    ignore_words = ['?', '!', '.', ',']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))  # remove duplicates and sort
    tags = sorted(set(tags))

    # Training data
    X_train = []
    y_train = []

    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)

        label = tags.index(tag)
        y_train.append(label)

    return np.array(X_train), np.array(y_train), all_words, tags
