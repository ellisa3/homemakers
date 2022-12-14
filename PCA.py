import numpy as np
import matplotlib.pyplot as plt
import wordembedding as we
from sklearn.decomposition import PCA
from config import fp

### Replicates Figure 6, a variance explained graph to show the components of PCA which hold the most variance in our dataset
### Method: took the difference of 10 gender pairs found in Figure 5 and performed PCA to identify that the gender component is the most importnt component
### Did a centering of the dataset not found in the paper but on their github

embedding = we.WordEmbedding(fp)

input_words = [('she','he'),('her','his'),('woman','man'),  ('herself', 'himself'), ('daughter', 'son'), ('mother', 'father'), ('gal','guy'),('girl','boy'),('female','male'), ("Mary", "John")] #Ten gendered pairs from figure 5 ('mary', 'john') removed bc not in our dataset,

pca = embedding.doPCA(input_words)


variance = pca.explained_variance_ratio_

cumulative_exp_var = np.cumsum(variance)


plt.bar(range(10), variance, align='edge', 
        label='Individual explained variance')

plt.step(range(10), cumulative_exp_var, where='mid',
         label='Cumulative explained variance', color='red') # for red line that shows cumulative sum
plt.ylabel('Explained variance percentage')
plt.xlabel('Principal components')
plt.xticks(ticks=np.arange(10,step=2))
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("/Users/darrylyork3/Desktop/Comps22/homemakers/visualizations/ExplainedVariancePlot.svg", format='svg')
# plt.show()