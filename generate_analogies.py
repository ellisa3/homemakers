#generating anologies
#a:x as b:y
# seed pair = [she, he] #[a,b]
# x = [] #literally every word in embedding?
# y = [] #only those who pass the threshold

#pseudocode
# !git remote set-url origin https://ellisa3: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@github.com/ellisa3/homemakers.git

from random import sample, seed
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile

class WordEmbedding:

    seedSimilarity = 0

    def __init__(self, fp):# -> None:
        # glove_file = datapath(fp)
        # word2vec_glove_file = get_tmpfile("w2v_gnews_small.txt") 
        # glove2word2vec(glove_file, word2vec_glove_file) 
        # self.model = KeyedVectors.load_word2vec_format(word2vec_glove_file, binary=False)
        self.model = KeyedVectors.load_word2vec_format(fp, binary=True)
        self.model.save_word2vec_format('/content/homemakers/data/GoogleNews-vectors-negative300.txt', binary=False)
        

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
                        print(word)
                        min_difference = curr_difference                  
                        min_word = word                                 #keep track of the [word, degree] that minimizes that difference
                analogy = (x, min_word[0], theta)                       #e.g., (homemakers, computer_programmer, 0.635)
                analogies.append(analogy)                               #[(homemakers, computer_programmer, 0.635), (nurse, doctor, 0.610), ...]
        f.close()
        return analogies
        
def main():
    fp = '/content/homemakers/data/GoogleNews-vectors-negative300.bin'
    we = WordEmbedding(fp)
    print(we.findSeedSimilarity())

    analogies = we.generateAnalogies('/mnt/c/CS/homemakers/data/small_x.txt')
    print(analogies)
    i = 0
    while (i < 8):
        print(analogies[i])
        i += 1

main()