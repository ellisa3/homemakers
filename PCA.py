import numpy as np
import matplotlib.pyplot as plt
from wordembedding import WordEmbedding
from sklearn.decomposition import PCA
from config import fp

### Replicates Figure 6, a variance explained graph to show the components of PCA which hold the most variance in our dataset
## fp should be w2v.txt file 

we = WordEmbedding(fp)
# Try w2v & our code

pca = PCA(n_components=None)

input_words = [('she','he'),('her','his'),('woman','man'), ('daughter', 'son'), ('mother', 'father'),
                ('gal','guy'),('girl','boy'),('female','male')] #Ten gendered pairs from figure 5 
input_vectors = []

for word in input_words:
    input_vectors.append(we.model[word[0]] - we.model[word[1]]) #Difference vectors of 10 gender pairs
    # input_word_vec1 - input_word_vec2
    # print("Pair: ",we.model[word[1]], we.model[word[0]])
input_vectors = np.array(input_vectors) # turn the word vectors into a np array
print(input_vectors)

data_std_pca = pca.fit(input_vectors) # fit data to model

variance = pca.explained_variance_ratio_ * 100
# print('Variance: ', variance)
cumulative_exp_var = np.cumsum(variance)
# print('cum_sum:', cumulative_exp_var)

plt.bar(range(len(variance)), variance, align='center', 
        label='Individual explained variance') # Top 2 components should be ~70% (Currently ~63%) 
        # 1) 38.33, 2) 62.921, 3) 17.46, 4) ~12, 5) ~7

plt.step(range(len(variance)), cumulative_exp_var, where='mid',
         label='Cumulative explained variance', color='red') # for red line that shows cumulative sum

plt.ylabel('Explained variance percentage')
plt.xlabel('Principal components')
plt.xticks(ticks=[]) # No ticks under bars
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig("visualizations/ExplainedVariancePlot.png")
plt.show()