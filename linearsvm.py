import numpy as np
from sklearn import svm
import json
import wordembedding
from config import fp_subset
from gensim.models import KeyedVectors
import sys


GENDER_SEED_INFILE = '/content/homemakers/data/gender_seed.json'
GS_OUTFILE = '/content/homemakers/data/Ogs_predict2.txt'
GN_OUTFILE = '/content/homemakers/data/Ogn_predict.txt'
gender_seed_words = set()

with open(GENDER_SEED_INFILE, "r") as f:
    gender_seed_words = json.load(f)

# file path of the embedding subset
# we_subset = wordembedding.WordEmbedding(fp_subset)
# with open("/content/homemakers/data/subset_words.txt", "w") as outfile:
#   for word in we_subset.model.index_to_key:
#     outfile.write(word)
#     outfile.write("\n")

# we = wordembedding.WordEmbedding(fp_subset2)
# with open("/content/homemakers/data/subset_words2.txt", "w") as outfile:
#   for word in we_subset.model.index_to_key:
#     outfile.write(word)
#     outfile.write("\n")

# sys.exit()

# file path of the full embedding
print("Loading embedding...")
we = wordembedding.WordEmbedding(isLinearSVM = True)


# gensim.models.keyedvectors.KeyedVectors(vector_size, count=0, dtype=<class 'numpy.float32'>, mapfile_path=None)
we_subset_list = []
with open("/content/homemakers/data/our_subset.txt", "w") as f:
  for i, word in enumerate(we.model.index_to_key):
    if i < 27000 or word in gender_seed_words:
      we_subset_list.append(word)
      f.write(word)
      f.write("\n")

print("Embedding has {} words.".format(len(we.model.index_to_key)))
print("{} seed words from '{}' out of which {} are in the embedding.".format(
    len(gender_seed_words),
    GENDER_SEED_INFILE,
    len([w for w in gender_seed_words if w in we.model.index_to_key]))
)

X = []
y = []
for word in we_subset_list:
  X.append(we.model[word])
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
          if word not in we_subset_list:
            test_value = np.array(we.model[word]).reshape((1, -1))
            y_pred = clf.predict(test_value)
            if y_pred == 1:
                gs_reader.write(word)
                gs_reader.write("\n")  
            elif y_pred == 0:
                gn_reader.write(word)
                gn_reader.write("\n")
