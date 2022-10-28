# from random import sample, seed
# from gensim.models import KeyedVectors
# from gensim.scripts.glove2word2vec import glove2word2vec
# from gensim.test.utils import datapath, get_tmpfile
import numpy as np
import wordembedding

# global we
# print(global we)
def data_load():
  global we 
  #we = wordembedding.WordEmbedding(fp = "/content/homemakers/data/w2v_gnews_small.txt", isLinearSVM = False)
  we = wordembedding.WordEmbedding(isLinearSVM = False)

data_load()

# print(we not None)

class GenerateAnalogies:
    def __init__(self):# -> None:
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
                break
                x = x.strip() #remove \r\n 
                # print("x:", x)
                # print("model[x]: ", type(self.model[x]))
                # print("model: ", type(self.model.vectors))
                differences = self.model[x] - self.model.vectors
                #np.savetxt('test.txt', differences)
                #print(differences)
                norms = np.linalg.norm(differences, axis=1)

                #print("norms: ", (norms))
                #print("index: ", self.model.index2word[0])
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
                analogies.append(analogy)                               #[(homemakers, computer_programmer, 0.635), (nurse, doctor, 0.610), ...]
        f.close()
        print(len(analogies))
        return analogies
        
def main():
    ga = GenerateAnalogies()

    ga.findSeedSimilarity()

    # #print(ga.seedSimilarity)
    # analogies = ga.generateAnalogies('/content/homemakers/data/small_x.txt')
    # i = 0
    # # print(analogies[i])

    # while (i < 4):
    #     print(analogies[i])
    #     i += 1


main()