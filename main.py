import numpy as np
import pandas as pd
import csv
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

data = pd.read_csv("train.csv")
labels = data['MaterialType'].values.flatten()

data = data.drop(['ID','CheckoutMonth','CheckoutYear','Checkouts','PublicationYear','MaterialType'], axis = 1)
train_text = data['CheckoutType'].map(str)+' '+data['UsageClass'].map(str)+' '+data['Title'].map(str)+' '+data['Creator'].map(str)+' '+data['Subjects'].map(str)+' '+data['Publisher'].map(str)
del data

le = preprocessing.LabelEncoder()
le.fit(labels)
train_labels = le.transform(labels)
del labels


data = pd.read_csv("test.csv")

data = data.drop(['ID','CheckoutMonth','CheckoutYear','Checkouts','PublicationYear'], axis = 1)
test_text = data['CheckoutType'].map(str)+' '+data['UsageClass'].map(str)+' '+data['Title'].map(str)+' '+data['Creator'].map(str)+' '+data['Subjects'].map(str)+' '+data['Publisher'].map(str)
del data

all_text = train_text
all_text.append(test_text)

sw = stopwords.words('english')
ss = SnowballStemmer('english')

corpus = []
for row in all_text:
    row = row.lower()
    row = row.split()
    row = [ss.stem(word) for word in row if not word in set(sw)]
    row = ' '.join(row)
    corpus.append(row)


vectorizer = CountVectorizer().fit(corpus)
train_features = vectorizer.transform(corpus)

forest = IsolationForest(n_estimators=100, contamination=0.045)
forest.fit(train_features)
outliers = forest.predict(train_features)

out = []
for i in range(0,len(outliers)):
    if outliers[i]==1:
        out.append(i)

corpus_in = []
train_labels_in = []

for i in out:
    corpus_in.append(corpus[i])
    train_labels_in.append(train_labels[i])

vectorizer = CountVectorizer().fit(corpus_in)
train_features_in = vectorizer.transform(corpus_in)
test_features = vectorizer.transform(test_text)

print(np.shape(train_features_in))
#X_train, X_test, Y_train, Y_test = train_test_split(train_features_in, train_labels_in, test_size = 0.2, random_state = 23)

clf = XGBClassifier(learning_rate=0.965)
#clf.fit(X_train,Y_train)
clf.fit(train_features_in, train_labels_in)

pred = clf.predict(test_features)
#pred = clf.predict(X_test)

#acc = accuracy_score(pred, Y_test)
#print(acc)

writer = csv.writer(open("result.csv", "w"))
head = ["ID", "MaterialType"]
writer.writerows([head])
cnt = 31654
for row in pred:
    writer.writerows([[str(cnt), str(le.inverse_transform(row))]])
    cnt = cnt+1

print(np.shape(train_features_in))








