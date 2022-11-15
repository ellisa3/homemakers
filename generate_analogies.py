from io import StringIO
import numpy as np
import wordembedding
import json

def data_load():
  global we 
  we = wordembedding.WordEmbedding(fp = "/content/homemakers/data/w2v_gnews_vsmall.txt", isLinearSVM = False)

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
        a = "she"
        b = "he"
        self.seedDirection = self.model[a] - self.model[b]
        #print("he/she: " , self.seedDirection)
        #print("she/he: ", self.model[b] - self.model[a])
        return self.seedDirection
        
    def generateAnalogies(self, filename):        
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
    analogies = ga.generateAnalogies('/content/homemakers/data/before_x.txt')
    i = 0
    print(analogies[i])
    f = open('data/beforeAnalogies.txt', 'w')
    i = 0
    for analogy in analogies:
      f.write(' '.join(analogy))
      f.write("\n")

    #run debiasing on the word embedding
    f = open("data/genderedPaper.json", 'r')
    gender_neutral = json.load(f)
    ga.we.debias(gender_neutral)

    #create analogies using the debiased word embedding
    analogies = ga.generateAnalogies('/content/homemakers/data/after_x.txt')
    i = 0
    print(analogies[i])
    f = open('data/afterAnalogies.txt', 'w')
    i = 0
    for analogy in analogies:
      f.write(analogy)
      f.write("\n")

main()