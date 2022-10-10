from random import sample
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile


class wordEmbedding:

    def __init__(self, fp) -> None:
        glove_file = datapath(fp)
        word2vec_glove_file = get_tmpfile("w2v_gnews_small.txt")
        glove2word2vec(glove_file, word2vec_glove_file)
        self.model = KeyedVectors.load_word2vec_format(word2vec_glove_file, binary=False)
    
    def generateOneSimilar(self, sampleWord):
        result = self.model.similar_by_word(sampleWord)
        most_similar_key, similarity = result[0]  # look at the first match
        return (most_similar_key, similarity)

    
    

def main():
    fp = '/Users/aldopolanco/homemakers/w2v_gnews_small.txt'
    we = wordEmbedding(fp)
    #Generate closest word and its similarities
    print(we.generateSimilar("man"))
    #Access a token's vector
    print(we.model["man"]) 
    #Access list of tokens in our word embedding
    #print(we.model.index_to_key)

        
main()