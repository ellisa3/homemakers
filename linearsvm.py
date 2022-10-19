import numpy as np
from sklearn import svm
import json
import wordembedding
from config import fp_subset, fp 

gender_seed_words = set()
with open('./data/gender_specific_seed_words.json', "r") as f:
    gender_seed_words = json.load(f)

# file path of the embedding subset
we_subset = wordembedding.WordEmbedding(fp_subset)
# file path of the full embedding
we = wordembedding.WordEmbedding(fp)

X = []
y = []
for word in we_subset.model.index_to_key:
    X.append(we_subset.getItem(word))
    if word not in gender_seed_words:
        y.append(1) # gender_neutral
    else: 
        y.append(0) # gender_specific
        
X=np.array(X)
y=np.array(y)

c = 1.0
clf = svm.LinearSVC(C = c)
clf.fit(X, y)

with open('./data/gender_specific_predict.txt', "w") as gs_reader:
    with open('./data/gender_neutral_predict.txt', "w") as gn_reader:
        for word in we.model.index_to_key:
            test_value = np.array(we.getItem(word)).reshape((1, -1))
            y_pred = clf.predict(test_value)
            if y_pred == 0:
                gs_reader.write(word)
                gs_reader.write("\n")  
            elif y_pred == 1:
                gn_reader.write(word)
                gn_reader.write("\n")