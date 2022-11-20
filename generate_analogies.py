from io import StringIO
import numpy as np
import wordembedding
import json
import time
import linearsvm

def data_load():
  global we 
  we = wordembedding.WordEmbedding(fp = "/content/homemakers/data/w2v_gnews_small.txt", isLinearSVM = False)

data_load()

class GenerateAnalogies:
    def __init__(self):
        self.seedDirection = 0
        self.we = we
        self.model = we.model
        self.GS = set()
        self.GN = set()

    def getScore(analogy):  #<< i feel like we can delete this, anyone use this?
        return analogy[2]

        
    #returns the difference vector of the seed pair (she, he)
    def findSeedSimilarity(self):
        a = "she"
        b = "he"
        self.seedDirection = self.model[a] - self.model[b]

        return self.seedDirection

    def generateGenderNeutralWords(self, gender_specific_fp):
        self.GS = list(set(line.strip() for line in open(gender_specific_fp)))
        for w in we.model.index_to_key:
            if w not in self.GS:
                self.GN.add(w)

    #returns an array of analogies generated from the word embedding
    def generateAnalogies(self):        
        analogies = []
        j = 0
        for w in self.model.index_to_key:
            x = self.model.index_to_key[j] 
            differences = self.model[x] - self.model.vectors
            norms = np.linalg.norm(differences, axis=1) 
            i = 0 
            maxScore = 0
            maxIndex = 0
            for norm in norms:
                if (norm <= 1):     #only include if ||x-y|| <= 1
                    score = np.dot(self.seedDirection, differences[i])
                    if (score > maxScore):    
                        maxScore = score 
                        maxIndex = i
                i += 1
            y = self.model.index_to_key[maxIndex] 

            if (isinstance(maxScore, np.float32)):
                maxScore = maxScore.astype(str)

            if (isinstance(maxScore, int)):
                maxScore = str(maxScore)
                
            analogy = [x, y, maxScore]
            analogies.append(analogy)  
            j += 1                         
        analogies.sort(reverse=True, key=lambda analogy: analogy[2])
        return analogies
        
def main():
    ga = GenerateAnalogies()
    ga.findSeedSimilarity()

    #create analogies using word embedding without debiasing
    analogies = ga.generateAnalogies()

    f = open('data/analogies_predebias.txt', 'w')
    for analogy in analogies:
      f.write(' '.join(analogy))
      f.write("\n")
    f.close()

    svm = linearsvm.LinearSVM()
    svm.generateGenderSpecificWords()
    ga.generateGenderNeutralWords("./data/our_GS_words.txt")

    #debias the word embedding
    ga.we.debias(ga.GN)

    #create analogies using the debiased word embedding
    analogies = ga.generateAnalogies()

    f = open('data/analogies_debias.txt', 'w')
    for analogy in analogies:
      f.write(' '.join(analogy))
      f.write("\n")
    f.close()

main()