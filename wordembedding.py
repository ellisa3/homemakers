from random import sample
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from sklearn.decomposition import PCA
import numpy as np
import gensim.downloader as api
from config import fp
import json


class WordEmbedding:

    def __init__(self, fp=None, isLinearSVM=False):
        if fp == None:
            print(isLinearSVM)
            if isLinearSVM:
                model = api.load('word2vec-google-news-300')
                self.model = model
            else:
                model = api.load('word2vec-google-news-300')
                self.model = model.wv
        else:
            glove_file = datapath(fp)
            word2vec_glove_file = get_tmpfile("w2v_gnews_small.txt") 
            glove2word2vec(glove_file, word2vec_glove_file) 
            self.model = KeyedVectors.load_word2vec_format(word2vec_glove_file, binary=False)
            with open("./data/definition_pairs.json") as dpfile:
                self.definition_pairs = json.load(dpfile)
            with open("./data/equalize_pairs.json") as dpfile:
                self.equalize_pairs = json.load(dpfile)
            with open("./data/Scatterplot.json") as spfile:
                self.scatterplot = json.load(spfile)

    
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

    def debias(self, gendered):
        #Find gender direction
        direction = self.findBiasDirection()
        gendered = [word.lower() for word in gendered]

        #Loop over words and debias them if they're gendered
        for word in self.model.index_to_key:
            if word in gendered:
                continue
            self.model[word] = self.drop(self.model[word], direction)
        
        candidates = []
        for w1, w2 in self.equalize_pairs:
            candidates.append((w1.lower(), w2.lower()))
            candidates.append((w1.upper(), w2.upper()))
            candidates.append((w1.title(), w2.title()))
        for w1, w2 in candidates:
            if (w1 not in self.model) or (w2 not in self.model):
              continue
            y = self.drop((self.model[w1] + self.model[w2])/2, direction)
            z = np.sqrt(1 - np.linalg.norm(y)**2)
            if (self.model[w1] - self.model[w2]).dot(direction) < 0:
                z = -z
            self.model[w1] = z * direction + y
            self.model[w2] = -z * direction + y
            # mu = (self.model[w1] + self.model[w2])/2
            # muproj = self.project(mu, direction)
            # v = mu - muproj
            # w1proj = self.project(w1, direction)
            # w2proj = self.project(w2, direction)
            # self.model[w1] = v + (np.sqrt((1 - np.linalg.norm(v)**2)) * (w1proj - muproj)/np.linalg.norm(w1proj - muproj))
            # self.model[w2] = v + (np.sqrt((1 - np.linalg.norm(v)**2)) * (w2proj - muproj)/np.linalg.norm(w2proj - muproj))

        self.norm()
        return None
    
    def norm(self):
        for word in self.model.index_to_key:
            self.model[word] = self.model[word]/np.linalg.norm(self.model[word])
        return self.model

    def normOne(self, vector):
        return vector/np.linalg.norm(vector)
    
    def project(self, word, direction):
        if isinstance(word, str):
            return np.dot(self.model[word], direction) * direction
        else: 
            return np.dot(word, direction) * direction
    
    def drop(self, u, v):
        return u - v * u.dot(v) / v.dot(v)

    def findBiasDirection(self):
        toFit = []
        for w1, w2 in self.definition_pairs:
            if w1.lower() not in self.model and w1 not in self.model:
                print("PROBLEM:", w1)
                continue
            else:
                if w1.lower() in self.model and w1 not in self.model:
                    w1 = w1.lower()
            if w2.lower() not in self.model and w2 not in self.model:
                print("PROBLEM:", w2)
                continue
            else:
                if w2.lower() in self.model and w2 not in self.model:
                    w2 = w2.lower()
            #Find average between two vector pairs such as (man, woman) or (he, she)
            average = (self.model[w1] + self.model[w2])/2
            #Add the difference between the average of both vectors to list of vectors to do PCA with
            toFit.append(self.normOne(self.model[w1] - average))
            toFit.append(self.normOne(self.model[w2] - average))
        toFit = np.array(toFit)
        pca = PCA(10)
        pca.fit(toFit)
        return pca.components_[0]
        
    def doPCA(self, definition_pairs):
        toFit = []
        for w1, w2 in definition_pairs:
            if w1 not in self.model.key_to_index or w2 not in self.model.key_to_index:
                print('Word Not found:')
                continue
            #Find average between two vector pairs such as (man, woman) or (he, she)
            average = (self.model[w1] + self.model[w2])/2
            #Add the difference between the average of both vectors to list of vectors to do PCA with
            toFit.append(self.normOne(self.model[w1] - average))
            toFit.append(self.normOne(self.model[w2] - average))

        toFit = np.array(toFit)
        pca = PCA(n_components=10)
        pca.fit(toFit)
        return pca

def main():
    we = WordEmbedding("content/homemakers/data/w2v_gnews_vsmall.txt")
    # diff = we.normOne(we.model["handbag"] - we.model["she"])
    # softball = we.normOne(we.normOne(we.model["he"]) + diff)
    # print("diff:", we.model.similar_by_vector(diff, topn=3))
    # print("should be softball:", we.model.similar_by_vector(softball, topn=3))
    # print("woman:", we.model.distance("woman", "homemaker"))
    # print("man:", we.model.distance("man", "homemaker"))

    # specific = open("data/genderedPaper.json")
    # specificwords = json.load(specific)
    # #specificwords = [word.lower() for word in specificwords]
    # we.debias(specificwords)
    # print("NEUTRALIZED")
    # diff = we.normOne(we.model["handbag"] - we.model["she"])
    # softball = we.normOne(we.model["he"] + diff)
    # print("diff:", we.model.similar_by_vector(diff, topn=3))
    # print("should be softball:", we.model.similar_by_vector(softball, topn=3))
    # print("woman:", we.model.distance("woman", "homemaker"))
    # print("man:", we.model.distance("man", "homemaker"))



    
    print("Done")
    
#main()