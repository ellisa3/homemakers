import numpy as np
import wordembedding

def data_load():
  global we 
  we = wordembedding.WordEmbedding(isLinearSVM = False)

data_load()

class GenerateAnalogies:
    def __init__(self):
        self.seedDirection = 0
        self.model = we.model
        
    #returns the cosine similarity between a and b, she,he = 0.612995028496, 0.612995028496
    def findSeedSimilarity(self):
        a = "she"
        b = "he"
        self.seedDirection = self.model[a] - self.model[b]
        #print(self.seedDirection)
        print(self.model.index_to_key)

        return self.seedDirection

      
    def generateAnalogies(self, filename):        
        analogies = []
        min_word = "" #maybe instantiated these outside the loop
        theta = 0
        with open(filename, 'r') as f:
            words = f.readlines()
            #j = 0
            for x in words:
                # break
                x = x.strip() 
                differences = self.model[x] - self.model.vectors
                norms = np.linalg.norm(differences, axis=1)
                i = 0 #keeps track of index to link vector back to key
                maxScore = 0
                maxIndex = 0
                for norm in norms:
                  if (norm <= 1):                   #only include if ||x-y|| <= 1
                    #key = self.model.index2word(i)
                    score = np.dot(self.seedDirection, differences[i])
                    if (score > maxScore):    #keep track of biggest score value and associated vector
                      maxScore = score 
                      maxIndex = i
                  i += 1
                key = self.model.index2word[maxIndex]
                print("key: ", key)
                analogy = [x, key]
                print(analogy)
                analogies.append(analogy)                   
        f.close()
        print(len(analogies))
        return analogies
        
def main():
    ga = GenerateAnalogies()
    ga.findSeedSimilarity()

main()