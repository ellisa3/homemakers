import numpy as np
from sklearn import svm
import json
import wordembedding
from config import fp_subset


GENDER_SEED_INFILE = '/content/homemakers/data/gender_seed.json'
GS_OUTFILE = '/content/homemakerss/data/Ogs_predict.txt'
GN_OUTFILE = '/content/homemakers/data/Ogn_predict.txt'
gender_seed_words = set()

with open(GENDER_SEED_INFILE, "r") as f:
    gender_seed_words = json.load(f)

# file path of the embedding subset
we_subset = wordembedding.WordEmbedding(fp_subset)
# file path of the full embedding
print("Loading embedding...")
we = wordembedding.WordEmbedding(isLinearSVM = True)

print("Embedding has {} words.".format(len(we.model.index_to_key)))
print("{} seed words from '{}' out of which {} are in the embedding.".format(
    len(gender_seed_words),
    GENDER_SEED_INFILE,
    len([w for w in gender_seed_words if w in we.model.index_to_key]))
)

X = []
y = []
for word in we_subset.model.index_to_key:
    X.append(we_subset.model[word])
    if word not in gender_seed_words:
        y.append(0) # gender_neutral
    else: 
        y.append(1) # gender_specific
        
X=np.array(X)
print("x shape ", X.shape)
y=np.array(y)

c = 1.0
clf = svm.LinearSVC(C = c)
clf.fit(X, y)

with open(GS_OUTFILE, "w") as gs_reader:
    with open(GN_OUTFILE, "w") as gn_reader:
        for word in we.model.index_to_key:
            test_value = np.array(we.model[word]).reshape((1, -1))
            y_pred = clf.predict(test_value)
            if y_pred == 1:
                gs_reader.write(word)
                gs_reader.write("\n")  
            elif y_pred == 0:
                gn_reader.write(word)
                gn_reader.write("\n")
