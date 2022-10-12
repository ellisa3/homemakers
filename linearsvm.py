import sys
import numpy as np
# import word embeddings
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import json
import wordembedding
"""
Learn gender specific words
Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings
Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai
2016
"""
gender_seed_words = set()
with open('./data/gender_seed_small.json', "r") as f:
    gender_seed_words = json.load(f)



# figure out how to load the embedding + get vector training data from embedding
    # what is the kye value for the embedding, and what format is the key?


# test if our thing works by using junk data!
fp = '/Users/swapnavarma/Desktop/cs-labs/comps/homemakers/data/w2v_gnews_super_small.txt'
we = wordembedding.WordEmbedding(fp)
fp2 = '/Users/swapnavarma/Desktop/cs-labs/comps/homemakers/data/w2v_gnews_small.txt'
E = wordembedding.WordEmbedding(fp2)

# for word in gender_seed_words:
#     y.append(we.getItem(word))
#     # print(we.getItem(word))
#     # print("\n")
# print(y)

X = []
y = []
for word in we.model.index_to_key:
    X.append(we.getItem(word))
    if word not in gender_seed_words:
        y.append(1)
    else: 
        y.append(0)
        
X=np.array(X)
y=np.array(y)
# print(X)

# what function does train_test_split() have
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=109) # 70% training and 30% test

c = 1.0
    # this is the regularization paramater
    # The bigger the C, the more penalty SVM gets when it makes misclassification
clf = svm.LinearSVC(C = c)

    # clf to store trained model values, which are further used to predict value, based on the previously stored weights
clf.fit(X, y)
# test something: "witch"
test_value = np.array(E.getItem('witch')).reshape((1, -1))
# print(test_value)
y_pred = clf.predict(test_value)
if y_pred == 0:
    print("gender_specific")
elif y_pred == 1:
    print("gender_neutral")
else:
    "?"
# print("gender specific" if y_pred == 0)
# print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # used to train the model with the seed data set
# clf.predict(# for new set)


# use this data for E 