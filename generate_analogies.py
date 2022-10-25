# from random import sample, seed
# from gensim.models import KeyedVectors
# from gensim.scripts.glove2word2vec import glove2word2vec
# from gensim.test.utils import datapath, get_tmpfile
import wordembedding

# global we
# print(global we)
def data_load():
  global we 
  we = wordembedding.WordEmbedding()

data_load()

# print(we not None)

class GenerateAnalogies:
    def __init__(self):# -> None:
        self.seedSimilarity = 0
        self.model = we.model
        
    #returns the cosine similarity between a and b, she,he = 0.612995028496, 0.612995028496
    def findSeedSimilarity(self):
        a = "he"
        b = "she"
        most_similar = self.model.similar_by_word(a)
        for similar_word in most_similar:
            if similar_word[0] == b:
                self.seedSimilarity = similar_word[1]
                return self.seedSimilarity
        return -1

    #either call it for each x OR call it once and runs on list of x's, latter more efficient    
    def generateAnalogies(self, filename):
        analogies = []
        with open(filename, 'r') as f:
            words = f.readlines()
            j = 0
            for x in words:
                x = x.strip() #remiove \r\n 
                cosine_similarities = self.model.similar_by_word(x, topn=10000)
                curr_difference = 0
                min_difference = 10
                for word in cosine_similarities:
                    #print(word)
                    theta = word[1]
                    #print(self.seedSimilarity)
                    curr_difference = abs(self.seedSimilarity - theta)  #want to find how similar the distance values are to each other
                    if curr_difference < min_difference:
                        #print(word)
                        min_difference = curr_difference                  
                        min_word = word                                 #keep track of the [word, degree] that minimizes that difference
                analogy = (x, min_word[0], theta)                       #e.g., (homemakers, computer_programmer, 0.635)
                analogies.append(analogy)                               #[(homemakers, computer_programmer, 0.635), (nurse, doctor, 0.610), ...]
        f.close()
        return analogies
        
def main():
    ga = GenerateAnalogies()
    analogies = ga.generateAnalogies('/content/homemakers/data/small_x.txt')
    i = 0
    while (i < 8):
        print(analogies[i])
        i += 1


main()