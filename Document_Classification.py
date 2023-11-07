# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model, Model
from keras.layers import Flatten, LSTM, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Dropout, Activation, Input, Dense, concatenate
from keras.layers.embeddings import Embedding
 
## Plotly
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)

# Others
import re
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

from sklearn.manifold import TSNE

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup as soup
from nltk.stem.snowball import SnowballStemmer

wordnet_lemmatizer = WordNetLemmatizer()

positive_reviews = soup(open('Reviews/books/positive.review').read())
positive_reviews = positive_reviews.findAll('review_text')

positive_result = []
for review in positive_reviews:
    review = [line.rstrip() for line in review if len(line) > 2]
    positive_result.append(review)

negative_reviews = soup(open('Reviews/books/negative.review').read())
negative_reviews = negative_reviews.findAll('review_text')

negative_result = []
for review in negative_reviews:
    review = [line.rstrip() for line in review if len(line) > 2]
    negative_result.append(review)

np.random.shuffle(positive_result)
positive_result = positive_result[:len(negative_result)]

reviews = []
for i in range(len(positive_result)):
    review = positive_result[i][0].replace('\r', '').replace('\n', '')
    review = " ".join(review.split())
    reviews.append(review)
    
for i in range(len(negative_result)):
    review = negative_result[i][0].replace('\r', '').replace('\n', '')
    review = " ".join(review.split())
    reviews.append(review)
    
labels = np.concatenate((np.ones((1000), dtype='int64'),
                    np.zeros((1000), dtype='int64')),axis=0)

data = pd.concat([pd.DataFrame(reviews, columns=['Reviews']), 
                  pd.DataFrame(labels, columns=['Labels'])],axis=1)
    
data = data.sample(frac=1)

def clean_text(text):
    
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)
    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    ## Stemming
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    return text
# apply the above function to df['text']
data['Reviews'] = data['Reviews'].map(lambda x: clean_text(x))

reviews = data['Reviews'].tolist()

from sklearn.model_selection import train_test_split
traindata, testdata = train_test_split(data, test_size = 0.25, random_state = 101)

embeddings_index = dict()
f = open('glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

vocabulary_size = 20000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(traindata['Reviews'])
trainsequences = tokenizer.texts_to_sequences(traindata['Reviews'])
trainseqs = pad_sequences(trainsequences, maxlen=50)
testsequences = tokenizer.texts_to_sequences(testdata['Reviews'])
testseqs = pad_sequences(testsequences, maxlen=50)


''' LSTM model '''
model = Sequential()
model.add(Embedding(20000, 100, input_length=50))
model.add(LSTM(500, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])

## Fit the model
model.fit(trainseqs, np.array(traindata['Labels']), validation_split=0.4, 
          epochs=5)
y_pred = model.predict(testseqs).ravel()
y_pred = (y_pred > 0.5)
y_test = np.array(testdata['Labels'])

from sklearn.metrics import accuracy_score, precision_score, recall_score
ac = accuracy_score(y_test, y_pred)
ps = precision_score(y_test, y_pred)
rs = recall_score(y_test, y_pred)

# Word2Vec Data processing
from gensim.models import Word2Vec
model_ug_cbow = Word2Vec(data['Reviews'].tolist())
model_ug_sg = Word2Vec(data['Reviews'].tolist(), sg=1)

embeddings_index_w2v = {}
for w in model_ug_cbow.wv.vocab.keys():
    embeddings_index_w2v[w] = np.append(model_ug_cbow.wv[w],model_ug_sg.wv[w])
    
num_words = 20000
embedding_matrix_w2v = np.zeros((num_words, 200))
for word, i in tokenizer.word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index_w2v.get(word)
    if embedding_vector is not None:
        embedding_matrix_w2v[i] = embedding_vector

''' NN Model with Word2Vec word embeddings '''

model_ptw2v = Sequential()
e = Embedding(20000, 200, weights=[embedding_matrix_w2v], input_length=50, 
              trainable=False)
model_ptw2v.add(e)
model_ptw2v.add(Flatten())
model_ptw2v.add(Dense(256, activation='relu'))
model_ptw2v.add(Dense(1, activation='sigmoid'))
model_ptw2v.compile(loss='binary_crossentropy', optimizer='adamax', 
                    metrics=['accuracy'])
model_ptw2v.fit(trainseqs, np.array(traindata['Labels']), 
                validation_split=0.4, 
                epochs=5, batch_size=32, verbose=2)

y_pred_ptw2v = model_ptw2v.predict(testseqs).ravel()
y_pred_ptw2v = (y_pred_ptw2v > 0.5)

from sklearn.metrics import accuracy_score, precision_score, recall_score
ac_ptw2v = accuracy_score(y_test, y_pred_ptw2v)
ps_ptw2v = precision_score(y_test, y_pred_ptw2v)
rs_ptw2v = recall_score(y_test, y_pred_ptw2v)

''' LSTM model with Word2Vec word embeddings'''
model_LSTM_w2v = Sequential()
model_LSTM_w2v.add(Embedding(20000, 200, weights=[embedding_matrix_w2v], 
                          input_length=50, trainable=True))
model_LSTM_w2v.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2))
model_LSTM_w2v.add(Dense(1, activation='sigmoid'))
model_LSTM_w2v.compile(loss='binary_crossentropy', optimizer='adamax', 
                       metrics=['accuracy'])

model_LSTM_w2v.summary()

from keras.callbacks import ModelCheckpoint
filepath="LSTM_best_weights_W2V.{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')
model_LSTM_w2v.fit(trainseqs, np.array(traindata['Labels']), 
          validation_split=0.30, 
          batch_size=32, epochs=10, 
          callbacks = [checkpoint])

#Select the best model saved
loaded_LSTM_model = load_model('LSTM_best_weights_W2V.03-0.7689.hdf5')
loaded_LSTM_model.evaluate(x=testseqs, y=y_test)

y_pred_LSTM_1 = loaded_LSTM_model.predict(testseqs).ravel()
y_pred_LSTM_1 = (y_pred_LSTM_1 > 0.5)

from sklearn.metrics import accuracy_score, precision_score, recall_score
ac_LSTM_1 = accuracy_score(y_test, y_pred_LSTM_1)
ps_LSTM_1 = precision_score(y_test, y_pred_LSTM_1)
rs_LSTM_1 = recall_score(y_test, y_pred_LSTM_1)

''' CNN Model with Word2Vec word embeddings '''
rev_input = Input(shape=(50,), dtype='int32')

rev_encoder = Embedding(20000, 200, weights=[embedding_matrix_w2v], 
                          input_length=50, trainable=True)(rev_input)
bigram_branch = Conv1D(filters=100, kernel_size=2, padding='valid', 
                       activation='relu', strides=1)(rev_encoder)
bigram_branch = GlobalMaxPooling1D()(bigram_branch)
trigram_branch = Conv1D(filters=100, kernel_size=3, padding='valid', 
                        activation='relu', strides=1)(rev_encoder)
trigram_branch = GlobalMaxPooling1D()(trigram_branch)
fourgram_branch = Conv1D(filters=100, kernel_size=4, padding='valid', 
                         activation='relu', strides=1)(rev_encoder)
fourgram_branch = GlobalMaxPooling1D()(fourgram_branch)
merged = concatenate([bigram_branch, trigram_branch, fourgram_branch], axis=1)

merged = Dense(256, activation='relu')(merged)
merged = Dropout(0.2)(merged)
merged = Dense(64, activation='relu')(merged)
merged = Dropout(0.2)(merged)
merged = Dense(1, activation='tanh')(merged)
output = Activation('sigmoid')(merged)
model_cnn = Model(inputs=[rev_input], outputs=[output])
model_cnn.compile(loss='binary_crossentropy',
                  optimizer='adamax',
                  metrics=['accuracy'])
model_cnn.summary()

from keras.callbacks import ModelCheckpoint
filepath="CNN_best_weights_W2V.{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')
model_cnn.fit(trainseqs, np.array(traindata['Labels']), 
          validation_split=0.30, 
          batch_size=32, epochs=20, 
          callbacks = [checkpoint])

# Select the best model saved
loaded_CNN_model = load_model('CNN_best_weights_W2V.03-0.7467.hdf5')
loaded_CNN_model.evaluate(x=testseqs, y=y_test)

y_pred_CNN1 = loaded_CNN_model.predict(testseqs).ravel()
y_pred_CNN1 = (y_pred_CNN1 > 0.5)

from sklearn.metrics import accuracy_score, precision_score, recall_score
ac_CNN1 = accuracy_score(y_test, y_pred_CNN1)
ps_CNN1 = precision_score(y_test, y_pred_CNN1)
rs_CNN1 = recall_score(y_test, y_pred_CNN1)

''' Glove word embeddings data processing '''
embeddings_index_glv = dict()
f = open('glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index_glv[word] = coefs
f.close()

embedding_matrix_glv = np.zeros((vocabulary_size, 100))
for word, index in tokenizer.word_index.items():
    if index > vocabulary_size - 1:
        break
    else:
        embedding_vector = embeddings_index_glv.get(word)
        if embedding_vector is not None:
            embedding_matrix_glv[index] = embedding_vector

''' NN Model with Glove word embeddings '''
model_ptglv = Sequential()
e = Embedding(20000, 100, weights=[embedding_matrix_glv], input_length=50, 
              trainable=False)
model_ptglv.add(e)
model_ptglv.add(Flatten())
model_ptglv.add(Dense(256, activation='relu'))
model_ptglv.add(Dense(1, activation='sigmoid'))
model_ptglv.compile(loss='binary_crossentropy', optimizer='adamax', 
                    metrics=['accuracy'])
model_ptglv.fit(trainseqs, np.array(traindata['Labels']), 
                validation_split=0.4, 
                epochs=10, batch_size=32, verbose=2)

y_pred_ptglv = model_ptglv.predict(testseqs).ravel()
y_pred_ptglv = (y_pred_ptglv > 0.5)

from sklearn.metrics import accuracy_score, precision_score, recall_score
ac_ptglv = accuracy_score(y_test, y_pred_ptglv)
ps_ptglv = precision_score(y_test, y_pred_ptglv)
rs_ptglv = recall_score(y_test, y_pred_ptglv)

''' LSTM model with Glove word embeddings'''
model_LSTM_glv = Sequential()
model_LSTM_glv.add(Embedding(20000, 100, weights=[embedding_matrix_glv], 
                          input_length=50, trainable=True))
model_LSTM_glv.add(LSTM(500, dropout=0.2, recurrent_dropout=0.2))
model_LSTM_glv.add(Dense(1, activation='sigmoid'))
model_LSTM_glv.compile(loss='binary_crossentropy', optimizer='adamax', 
                       metrics=['accuracy'])

model_LSTM_glv.summary()

from keras.callbacks import ModelCheckpoint
filepath="LSTM_best_weights_GLV.{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')
model_LSTM_glv.fit(trainseqs, np.array(traindata['Labels']), 
          validation_split=0.30, 
          batch_size=32, epochs=10, 
          callbacks = [checkpoint])

# Select the best model saved
loaded_LSTM_model = load_model('LSTM_best_weights_GLV.10-0.7111.hdf5')
loaded_LSTM_model.evaluate(x=testseqs, y=y_test)

y_pred_LSTM_2 = loaded_LSTM_model.predict(testseqs).ravel()
y_pred_LSTM_2 = (y_pred_LSTM_2 > 0.5)

from sklearn.metrics import accuracy_score, precision_score, recall_score
ac_LSTM_2 = accuracy_score(y_test, y_pred_LSTM_2)
ps_LSTM_2 = precision_score(y_test, y_pred_LSTM_2)
rs_LSTM_2 = recall_score(y_test, y_pred_LSTM_2)

''' CNN Model with Glove word embeddings '''
rev_input = Input(shape=(50,), dtype='int32')

rev_encoder = Embedding(20000, 100, weights=[embedding_matrix_glv], 
                          input_length=50, trainable=True)(rev_input)
bigram_branch = Conv1D(filters=100, kernel_size=2, padding='valid', 
                       activation='relu', strides=1)(rev_encoder)
bigram_branch = GlobalMaxPooling1D()(bigram_branch)
trigram_branch = Conv1D(filters=100, kernel_size=3, padding='valid', 
                        activation='relu', strides=1)(rev_encoder)
trigram_branch = GlobalMaxPooling1D()(trigram_branch)
fourgram_branch = Conv1D(filters=100, kernel_size=4, padding='valid', 
                         activation='relu', strides=1)(rev_encoder)
fourgram_branch = GlobalMaxPooling1D()(fourgram_branch)
merged = concatenate([bigram_branch, trigram_branch, fourgram_branch], axis=1)

merged = Dense(256, activation='relu')(merged)
merged = Dropout(0.2)(merged)
merged = Dense(1)(merged)
output = Activation('sigmoid')(merged)
model_cnn2 = Model(inputs=[rev_input], outputs=[output])
model_cnn2.compile(loss='binary_crossentropy',
                  optimizer='adamax',
                  metrics=['accuracy'])
model_cnn2.summary()

from keras.callbacks import ModelCheckpoint
filepath="CNN_best_weights_GLV.{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')
model_cnn2.fit(trainseqs, np.array(traindata['Labels']), 
          validation_split=0.4, 
          batch_size=32, epochs=10, 
          callbacks = [checkpoint])

# Select the best model
loaded_CNN_model2 = load_model('CNN_best_weights_GLV.10-0.7083.hdf5')
loaded_CNN_model2.evaluate(x=testseqs, y=y_test)

y_pred_CNN2 = loaded_CNN_model2.predict(testseqs).ravel()
y_pred_CNN2 = (y_pred_CNN2 > 0.5)

from sklearn.metrics import accuracy_score, precision_score, recall_score
ac_CNN2 = accuracy_score(y_test, y_pred_CNN2)
ps_CNN2 = precision_score(y_test, y_pred_CNN2)
rs_CNN2 = recall_score(y_test, y_pred_CNN2)

''' Super CNN Model with Word2Vec and Glove word embeddings '''
rev_input = Input(shape=(50,), dtype='int32')

rev_encoder_1 = Embedding(20000, 100, weights=[embedding_matrix_glv], 
                          input_length=50, trainable=True)(rev_input)
bigram_branch_1 = Conv1D(filters=100, kernel_size=2, padding='valid', 
                       activation='relu', strides=1)(rev_encoder_1)
bigram_branch_1 = GlobalMaxPooling1D()(bigram_branch_1)
trigram_branch_1 = Conv1D(filters=100, kernel_size=3, padding='valid', 
                        activation='relu', strides=1)(rev_encoder_1)
trigram_branch_1 = GlobalMaxPooling1D()(trigram_branch_1)
fourgram_branch_1 = Conv1D(filters=100, kernel_size=4, padding='valid', 
                         activation='relu', strides=1)(rev_encoder_1)
fourgram_branch_1 = GlobalMaxPooling1D()(fourgram_branch_1)
cnn_glv = concatenate([bigram_branch_1, trigram_branch_1, fourgram_branch_1], axis=1)

rev_encoder_2 = Embedding(20000, 200, weights=[embedding_matrix_w2v], 
                          input_length=50, trainable=True)(rev_input)
bigram_branch_2 = Conv1D(filters=100, kernel_size=2, padding='valid', 
                       activation='relu', strides=1)(rev_encoder_2)
bigram_branch_2 = GlobalMaxPooling1D()(bigram_branch_2)
trigram_branch_2 = Conv1D(filters=100, kernel_size=3, padding='valid', 
                        activation='relu', strides=1)(rev_encoder_2)
trigram_branch_2 = GlobalMaxPooling1D()(trigram_branch_2)
fourgram_branch_2 = Conv1D(filters=100, kernel_size=4, padding='valid', 
                         activation='relu', strides=1)(rev_encoder_2)
fourgram_branch_2 = GlobalMaxPooling1D()(fourgram_branch_2)
cnn_w2v = concatenate([bigram_branch_2, trigram_branch_2, fourgram_branch_2], axis=1)

supercnn = concatenate([cnn_glv, cnn_w2v], axis=1)
supercnn = Dense(64, activation='tanh')(supercnn)
supercnn = Dropout(0.4)(supercnn)
supercnn = Dense(32, activation='relu')(supercnn)
supercnn = Dropout(0.35)(supercnn)
supercnn = Dense(1)(supercnn)
output = Activation('sigmoid')(supercnn)
supercnn = Model(inputs=[rev_input], outputs=[output])
supercnn.compile(loss='binary_crossentropy',
                  optimizer='adamax',
                  metrics=['accuracy'])
supercnn.summary()

from keras.callbacks import ModelCheckpoint
filepath="SuperCNN_best_weights.{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')
supercnn.fit(trainseqs, np.array(traindata['Labels']), 
          validation_split=0.3, 
          batch_size=32, epochs=20, 
          callbacks = [checkpoint])

# Select the best model
loaded_supercnn = load_model('SuperCNN_best_weights.16-0.7778.hdf5')
loaded_supercnn.evaluate(x=testseqs, y=y_test)

y_pred_supercnn = loaded_supercnn.predict(testseqs).ravel()
y_pred_supercnn = (y_pred_supercnn > 0.5)

from sklearn.metrics import accuracy_score, precision_score, recall_score
ac_supercnn = accuracy_score(y_test, y_pred_supercnn)
ps_supercnn = precision_score(y_test, y_pred_supercnn)
rs_supercnn = recall_score(y_test, y_pred_supercnn)

''' Logistic Regression Model with TfIdf '''
from sklearn.feature_extraction.text import TfidfVectorizer
tvec = TfidfVectorizer(max_features=5000)
tvec.fit(traindata['Reviews'])

x_train_tfidf = tvec.transform(traindata['Reviews'])
x_test_tfidf = tvec.transform(testdata['Reviews'])

lr_with_tfidf = LogisticRegression()
lr_with_tfidf.fit(x_train_tfidf,traindata['Labels'])

y_pred_lr = lr_with_tfidf.predict(x_test_tfidf)

from sklearn.metrics import accuracy_score, precision_score, recall_score
ac_lr = accuracy_score(y_test, y_pred_lr)
ps_lr = precision_score(y_test, y_pred_lr)
rs_lr = recall_score(y_test, y_pred_lr)