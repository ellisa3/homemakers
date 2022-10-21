from random import sample
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from sklearn.decomposition import PCA
import numpy as np
from config import fp
import json


class WordEmbedding:

    def __init__(self, fp):
        glove_file = datapath(fp)
        word2vec_glove_file = get_tmpfile("w2v_gnews_small.txt") 
        glove2word2vec(glove_file, word2vec_glove_file) 
        self.model = KeyedVectors.load_word2vec_format(word2vec_glove_file, binary=False)
        with open("./data/definition_pairs.json") as dpfile:
            self.definition_pairs = json.load(dpfile)
    
    def generateOneSimilar(self, sampleWord): #this function exists in keyedVectors, most_similar() (set param N to 1 to get most similar word) line 776 of documentation
        result = self.model.similar_by_word(sampleWord)
        most_similar_key, similarity = result[0]  # look at the first match
        return (most_similar_key, similarity)
    
    def generateNSimilar(self, input_words, num_sim):
        ## Generate a list of n similar words to a list of input words
        def append_list(sim_words, words):
            list_of_words = []
            
            for i in range(len(sim_words)): # Create tuple of words & similarity
                
                sim_words_list = list(sim_words[i])
                sim_words_list.append(words)
                sim_words_tuple = tuple(sim_words_list)
                list_of_words.append(sim_words_tuple)

            print("list of words", list_of_words)    
            return list_of_words
        
        user_input = [word.strip() for word in input_words.split(',')]
        result_words = []

        for words in user_input:
            sim_words = self.model.most_similar(words, topn = num_sim)
            sim_words = append_list(sim_words, words)
            result_words.extend(sim_words)

        # most_similar_keys = [word[0] for word in result_words]
        # similarity = [word[1] for word in result_words]
        # similar_to = [word[2] for word in result_words]
        return result_words

    def neutralize(self, gendered):
        #Find gender direction
        toFit = []
        for w1, w2 in self.definition_pairs:
            if w1 not in self.model or w2 not in self.model:
                continue
            w1 = w1.lower()
            w2 = w2.lower()
            #Find average between two vector pairs such as (man, woman) or (he, she)
            average = (self.model[w1] + self.model[w2])/2
            #Add the difference between the average of both vectors to list of vectors to do PCA with
            toFit.append(self.model[w1] - average)
            toFit.append(self.model[w2] - average)
        pca = PCA(1)
        pca.fit(toFit)
        direction = pca.components_[0]
        for word in self.model.index_to_key:
            if word in gendered:
                continue
            else:
                newvec = np.dot(self.model[word], direction) * direction
                newvec = newvec/np.linalg.norm(newvec)
                self.model[word] = newvec
        return None

            


def main():
    we = WordEmbedding(fp)
    print(we.generateNSimilar("nurse", 5))
    we.neutralize(["he", "she", "woman", "man"])
    print("NEUTRALIZED")
    print(we.generateNSimilar("nurse", 5))
    print("Done")
    
        
# main()