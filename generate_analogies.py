from io import StringIO
import numpy as np
import wordembedding
import json
import time

def data_load():
  global we 
  we = wordembedding.WordEmbedding(fp = "/content/homemakers/data/w2v_gnews_small.txt", isLinearSVM = False)

data_load()

class GenerateAnalogies:
    def __init__(self):
        self.seedDirection = 0
        self.we = we
        self.model = we.model

    def getScore(analogy):
        return analogy[2]

        
    #returns the cosine similarity between a and b, she,he = 0.612995028496, 0.612995028496
    def findSeedSimilarity(self):
        a = "he"
        b = "she"
        self.seedDirection = self.model[a] - self.model[b]
        #print("he/she: " , self.seedDirection)
        #print("she/he: ", self.model[b] - self.model[a])
        return self.seedDirection
        
    def generateAnalogies(self):        
        analogies = []
        j = 0
        for w in self.model.index_to_key:
            x = self.model.index_to_key[j] 
            differences = self.model[x] - self.model.vectors
            norms = np.linalg.norm(differences, axis=1) #<--error here
            i = 0 #keeps track of index to link vector back to key
            maxScore = 0
            maxIndex = 0
            for norm in norms:
                if (norm <= 1):     #only include if ||x-y|| <= 1
                    score = np.dot(self.seedDirection, differences[i])
                    if (score > maxScore):    #keep track of biggest score value and associated vector
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
        print("length: ", len(analogies))
        analogies.sort(reverse=True, key=lambda analogy: analogy[2])
        return analogies
        
def main():
    ga = GenerateAnalogies()
    ga.findSeedSimilarity()

    #create analogies using word embedding without debiasing
    start = time.time()
    analogies = ga.generateAnalogies()
    end = time.time()
    print("analogies before:" + str(end - start))
    i = 0
    # print(analogies[i])
    f = open('data/before_analogies', 'w')
    i = 0
    for analogy in analogies:
      #print(analogy)
      f.write(' '.join(analogy))
      f.write("\n")
    f.close()
    #run debiasing on the word embedding

    start = time.time()
    fp = open('/content/homemakers/data/gender_specific_full.json')
    gender_neutral = json.load(fp)
    end = time.time()
    print("svm: " + str(end - start))

    start = time.time()
    ga.we.debias(gender_neutral)
    end = time.time()
    print("debias: " + str(end - start))
    #create analogies using the debiased word embedding
    start = time.time()
    analogies = ga.generateAnalogies()
    end = time.time()
    print("analogies after:" + str(end - start))
    i = 0
    # print(analogies[i])
    f = open('data/he_she_after_analogies.txt', 'w')
    i = 0
    for analogy in analogies:
      #print(analogy)
      f.write(' '.join(analogy))
      #f.write(analogy[0] + ", " + analogy[1] + ", " + analogy[0])
      f.write("\n")
    f.close()
    print("done")

main()