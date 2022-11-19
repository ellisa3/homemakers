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

    def __init__(self, fp=None, isLinearSVM=False, we_subset_list=None, we_vector_list = None):
        if fp == None:
            print(isLinearSVM)
            if isLinearSVM:
                model = api.load('word2vec-google-news-300')
                self.model = model
            elif we_subset_list:
              self.model = KeyedVectors(300, 27000)
              self.model.add_vectors(we_subset_list, we_vector_list)
            else:
                model = api.load('word2vec-google-news-300')
                self.model = model.wv
        else:
            glove_file = datapath(fp)
            word2vec_glove_file = get_tmpfile("w2v_paper.txt") 
            glove2word2vec(glove_file, word2vec_glove_file) 
            self.model = KeyedVectors.load_word2vec_format(word2vec_glove_file, binary=False)
            with open("./data/definition_pairs.json") as dpfile:
                self.definition_pairs = json.load(dpfile)
            with open("./data/equalize_pairs.json") as dpfile:
                self.equalize_pairs = json.load(dpfile)
            with open("./data/Scatterplot.json") as spfile:
                self.scatterplot = json.load(spfile)

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
                continue
            else:
                if w1.lower() in self.model and w1 not in self.model:
                    w1 = w1.lower()
            if w2.lower() not in self.model and w2 not in self.model:
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
