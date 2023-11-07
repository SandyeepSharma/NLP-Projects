import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
#nltk.download('wordnet')

df = pd.read_csv('bbc-text.csv')
df['category'].value_counts()

stemmer = SnowballStemmer("english")

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

processed_docs = df['text'].map(preprocess)
processed_docs[:10]

corpus = []
for i in range(len(processed_docs)):
    article = processed_docs[i]
    article = ' '.join(article)
    corpus.append(article)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', 
                        ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(corpus).toarray()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(features)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_val_score


models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, y_kmeans, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

cv_df.groupby('model_name').accuracy.mean()

from sklearn.model_selection import train_test_split

model = LogisticRegression(random_state=0)

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, y_kmeans, df.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)


from sklearn.metrics import confusion_matrix
import seaborn as sns

conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=np.unique(y_kmeans), yticklabels=np.unique(y_kmeans))
plt.ylabel('Actual')
plt.xlabel('Predicted')

topics={}
for i in range(model.coef_.shape[0]):
    topics[i] =  model.coef_[i]
    
topic_features = pd.DataFrame()
vocab = tfidf.get_feature_names()

for topic, coeffs in topics.items():
    topic_features = pd.concat([pd.DataFrame(vocab, columns=['Vocab']), 
                                pd.DataFrame(coeffs, columns=['Importances'])], axis=1)
    topic_features = topic_features.sort_values('Importances', 
                                                ascending=['False']).head(50)
    topic_features.to_csv('Feature_Impotance_{}'.format(topic), index=False)