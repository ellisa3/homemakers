import numpy as np
import wordembedding

def data_load():
  global we 
  we = wordembedding.WordEmbedding(fp = "/content/homemakers/data/w2v_gnews_small.txt", isLinearSVM = False)

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
        #print("seedDirection: ", self.seedDirection)
        #print(self.model.index_to_key)

        return self.seedDirection

      
    def generateAnalogies(self, filename):        
        analogies = []
        with open(filename, 'r') as f:
            words = f.readlines()
            for x in words:
                x = x.strip() #remove \r\n 
                # print("x:", x)
                # print("model[x]: ", type(self.model[x]))
                # print("model: ", type(self.model.vectors))
                differences = self.model[x] - self.model.vectors
                #np.savetxt('test.txt', differences)
                norms = np.linalg.norm(differences, axis=1) #<--error here
                i = 0 #keeps track of index to link vector back to key
                maxScore = 0
                maxIndex = 0
                for norm in norms:
                  #print("norm: ", norm)
                  if (norm <= 1):                   #only include if ||x-y|| <= 1
                    #print("norm <= 1")          
                    score = np.dot(self.seedDirection, differences[i])
                    if (score > maxScore):    #keep track of biggest score value and associated vector
                      maxScore = score 
                      maxIndex = i
                  i += 1
                key = self.model.index2word[maxIndex] #</s>
                #print("key: ", key)
                analogy = [x, key]
                #print(analogy)
                analogies.append(analogy)                               #[(homemakers, computer_programmer, 0.635), (nurse, doctor, 0.610), ...]
        f.close()
        print("length: ", len(analogies))
        return analogies
        
def main():
    ga = GenerateAnalogies()
    ga.findSeedSimilarity()

    # #print(ga.seedSimilarity)
    analogies = ga.generateAnalogies('/content/homemakers/data/before_x.txt')
    i = 0
    print(analogies[i])
    f = open('we_analogies.txt', 'a')
    i = 0
    for analogy in analogies:
      f.write(" ".join(analogy))
      f.write("\n")

    # while (i < 4):
    #     print(analogies[i])
    #     i += 1


main()