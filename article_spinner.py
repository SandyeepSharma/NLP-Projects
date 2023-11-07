#from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import nltk
import random
import numpy as np

from bs4 import BeautifulSoup


# load the reviews
# data courtesy of http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html
positive_reviews = BeautifulSoup(open('Reviews/dvd/positive.review').read())
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open('Reviews/dvd/negative.review').read())
negative_reviews = negative_reviews.findAll('review_text')

neutral_reviews = BeautifulSoup(open('Reviews/dvd/unlabeled.review').read())
neutral_reviews = neutral_reviews.findAll('review_text')

reviews = []
for review in positive_reviews:
    review = [line.rstrip() for line in review if len(line) > 2]
    reviews.append(review)
    
for review in negative_reviews:
    review = [line.rstrip() for line in review if len(line) > 2]
    reviews.append(review)
    
for review in neutral_reviews:
    review = [line.rstrip() for line in review if len(line) > 2]
    reviews.append(review)

# extract trigrams and insert into dictionary
# (w1, w3) is the key, [ w2 ] are the values
type(reviews[0][0])
trigrams = {}
for review in reviews:
    s = review[0].lower()
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens) - 2):
        k = (tokens[i], tokens[i+2])
        if k not in trigrams:
            trigrams[k] = []
        trigrams[k].append(tokens[i+1])

# turn each array of middle-words into a probability vector
trigram_probabilities = {}
for k, words in trigrams.items():
    # create a dictionary of word -> count
    if len(set(words)) > 1:
        # only do this when there are different possibilities for a middle word
        d = {}
        #n = len(words)
        n = 0
        for w in words:
            if w not in d:
                d[w] = 0
            d[w] += 1
            n += 1
        for w, c in d.items():
            d[w] = float(c) / n
        trigram_probabilities[k] = d

def random_sample(d):
    # choose a random sample from dictionary where values are the probabilities
    r = random.random()
    cumulative = 0
    for w, p in d.items():
        cumulative = cumulative + p
        if r < cumulative:
            return w


def test_spinner():
    review = random.choice(reviews)
    s = review[0].lower()
    print("Original:", s)
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens) - 2):
        if random.random() < 0.2: # 20% chance of replacement
            k = (tokens[i], tokens[i+2])
            if k in trigram_probabilities:
                w = random_sample(trigram_probabilities[k])
                tokens[i+1] = w
    print("Spun:")
    print(" ".join(tokens).replace(" .", ".").replace(" '", "'").replace(" ,", ",").replace("$ ", "$").replace(" !", "!"))


if __name__ == '__main__':
    test_spinner()