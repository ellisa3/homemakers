import numpy as np
import wordembedding
import json

def data_load():
  global we 
  we = wordembedding.WordEmbedding(fp = "/content/homemakers/data/w2v_gnews_small.txt", isLinearSVM = False)

data_load()

class GenerateAnalogies:
    def __init__(self):
        self.seedDirection = 0
        self.we = we
        self.model = we.model
        
    #returns the cosine similarity between a and b, she,he = 0.612995028496, 0.612995028496
    def findSeedSimilarity(self):
        a = "he"
        b = "she"
        self.seedDirection = self.model[a] - self.model[b]
        print("he/she: " , self.seedDirection)
        print("she/he: ", self.model[b] - self.model[a])


        return self.seedDirection

      
    def generateAnalogies(self, filename):        
        analogies = []
        with open(filename, 'r') as f:
            words = f.readlines()
            for x in words:
                x = x.strip() #remove \r\n 
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
                key = self.model.index_to_key[maxIndex] 
                analogy = [x, key]
                analogies.append(analogy)                             
        f.close()
        print("length: ", len(analogies))
        return analogies
        
def main():
    ga = GenerateAnalogies()
    ga.findSeedSimilarity()

    analogies = ga.generateAnalogies('/content/homemakers/data/parsed_occupations.txt')
    i = 0
    print(analogies[i])
    f = open('bias_analogies.txt', 'a')
    i = 0
    for analogy in analogies:
      f.write(" ".join(analogy))
      f.write("\n")

    # gender_neutral = []
    # f = open("data/gender_neutral_predict.txt", 'r')
    # words = f.readlines()
    # for word in words:
    #     word = word.strip() #remove \r\n 
    #     gender_neutral.append(word)

    # f = open("data/genderedPaper.json", 'r')
    # gender_neutral = json.load(f)
    # ga.we.debias(gender_neutral)


    # analogies = ga.generateAnalogies('/content/homemakers/data/parsed_occupations.txt')
    # i = 0
    # print(analogies[i])
    # f = open('debias_analogies.txt', 'a')
    # i = 0
    # for analogy in analogies:
    #   f.write(" ".join(analogy))
    #   f.write("\n")

main()