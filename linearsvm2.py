
from __future__ import print_function, division
import sys
import argparse
import wordembedding
import numpy as np
from sklearn.svm import LinearSVC
import json
if sys.version_info[0] < 3:
    import io
    open = io.open
"""
Learn gender specific words
Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings
Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai
2016
"""

parser = argparse.ArgumentParser()
parser.add_argument("NUM_TRAINING", type=int)

args = parser.parse_args()

NUM_TRAINING = args.NUM_TRAINING
GENDER_SPECIFIC_SEED_WORDS = "/content/homemakers/data/gender_seed.json"
OUTFILE = "/content/homemakers/data/Tgs_predict.txt"

with open(GENDER_SPECIFIC_SEED_WORDS, "r") as f:
    gender_seed = json.load(f)

print("Loading embedding...")
E = wordembedding.WordEmbedding(isLinearSVM = True)

print("Embedding has {} words.".format(len(E.model.index_to_key)))
print("{} seed words from '{}' out of which {} are in the embedding.".format(
    len(gender_seed),
    GENDER_SPECIFIC_SEED_WORDS,
    len([w for w in gender_seed if w in E.model.index_to_key]))
)

gender_seed = list(set(w for i, w in enumerate(E.model.index_to_key) if w in gender_seed or (w.lower() in gender_seed and i<NUM_TRAINING)))
with open("their_gender_seed.json", "w") as f:
  json.dump(gender_seed, f)
sys.exit()
# i=0
# for w in enumerate(E.model.index_to_key):
#   new_val = 1 if w in gender_seed else 0
#   print(new_val)
#   if (i<NUM_TRAINING or w in gender_seed):
#     if new_val == 1:
#       print(w)
#   i+=1
# sys.exit()

labeled_train = [(i, 1 if w in gender_seed else 0) for i, w in enumerate(E.model.index_to_key) if (i<NUM_TRAINING or w in gender_seed)]
train_indices, train_labels = zip(*labeled_train)
y = np.array(train_labels)
print(y.sum())


vecs = np.array([E.model[w] for w in E.model.index_to_key])
X = np.array([vecs[i] for i in train_indices])
# np.savetxt("X_theirs.txt", X)
# np.savetxt("y_theirs.txt", y)

C = 1.0
clf = LinearSVC(C=C, tol=0.0001)
clf.fit(X, y)
np.savetxt("clf_coef_theirs.txt", clf.coef_)
np.savetxt("clf_intercept_theirs.txt", clf.intercept_) 


sys.exit()


print("x shape ", X.shape)
weights = (0.5 / (sum(y)) * y + 0.5 / (sum(1 - y)) * (1 - y))
weights = 1.0 / len(y)
score = sum((clf.predict(X) == y) * weights)
# print(1 - score, sum(y) * 1.0 / len(y))

# pred = clf.coef_[0].dot(X.T)
# direction = clf.coef_[0]
# intercept = clf.intercept_
print("here")

# .dot(a, b)
# Tndarray of shape (n_samples, n_classes)
vecs_len = len(vecs)
v_1_i = vecs_len // 5
v_2_i = (vecs_len //5) * 2
v_3_i = (vecs_len //5) * 3
v_4_i = (vecs_len //5) * 4

vecs_one = vecs[0:v_1_i]
vecs_two = vecs[v_1_i:v_2_i]
vecs_three = vecs[v_2_i:v_3_i]
vecs_four = vecs[v_3_i:v_4_i]
vecs_five = vecs[v_4_i:(vecs_len + 1)]
print("here1")


is_gender_specific1 = (vecs_one.dot(clf.coef_.T) > -clf.intercept_)
is_gender_specific2 = (vecs_two.dot(clf.coef_.T) > -clf.intercept_)
is_gender_specific3 = (vecs_three.dot(clf.coef_.T) > -clf.intercept_)
is_gender_specific4 = (vecs_four.dot(clf.coef_.T) > -clf.intercept_)
is_gender_specific5 = (vecs_five.dot(clf.coef_.T) > -clf.intercept_)
print("here2")

full_gender_specific1 = list(set([w for label, w in zip(is_gender_specific1, E.model.index_to_key)
                            if label]).union(gender_seed))

full_gender_specific2 = list(set([w for label, w in zip(is_gender_specific2, E.model.index_to_key)
                            if label]).union(gender_seed))

full_gender_specific3 = list(set([w for label, w in zip(is_gender_specific3, E.model.index_to_key)
                            if label]).union(gender_seed))

full_gender_specific4 = list(set([w for label, w in zip(is_gender_specific4, E.model.index_to_key)
                            if label]).union(gender_seed))

full_gender_specific5 = list(set([w for label, w in zip(is_gender_specific5, E.model.index_to_key)
                            if label]).union(gender_seed))

print("here3")
full_gender_specific = np.array(full_gender_specific1+full_gender_specific2+full_gender_specific3+full_gender_specific4+full_gender_specific5)
print(type(full_gender_specific))
# print(len(full_gender_specific1) + len(full_gender_specific2) + len(full_gender_specific3) + len(full_gender_specific4) + len(full_gender_specific5))

with open(OUTFILE, "a") as f:
    json.dump(full_gender_specific, f)
