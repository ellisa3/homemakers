#generating anologies
#a:x as b:y
# seed pair = [she, he] #[a,b]
# x = [] #literally every word in embedding?
# y = [] #only those who pass the threshold

#pseudocode

# find cosine similarity between she, he
# she_he_similarity = most_similar(positive = [she], topn=10)      //solid guess that given she, he will be within top 10 most similar words
# for every word in x
    # cosine_similarities = []
    # cosine_similarities = most_similar(positive = [x], topn=None) //returns all the words in the vocab & their cosine similarity relative to x

    # curr_difference = 0
    # min_difference = 10
    # min_theta = 0                                             //not sure what type this boi is
    # for theta in cosine_similarities
        # curr_difference = abs_val(she_he_similarity - theta)  //want to find how similar the distance values are to each other
        # if curr_difference < min_difference
            # min_difference = curr_difference                  
            # min_theta = theta                                 //keep track of the [word, degree] that minimizes that difference

    #return min_theta

from random import sample
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile

    # return cosine similarity between she, he

        
