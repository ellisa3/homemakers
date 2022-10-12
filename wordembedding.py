from random import sample
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile


class wordEmbedding:

    def __init__(self, fp): # -> None:
        glove_file = datapath(fp)
        word2vec_glove_file = get_tmpfile("w2v_gnews_small.txt")
        glove2word2vec(glove_file, word2vec_glove_file)
        self.model = KeyedVectors.load_word2vec_format(word2vec_glove_file, binary=False)
        print(self.model)

    def generateSimilar(self, sampleWord):
        result = self.model.similar_by_word(sampleWord)
        most_similar_key, similarity = result[0]  # look at the first match
        return (most_similar_key, similarity)


def main():
    fp = '/mnt/c/CS/homemakers/w2v_gnews_small.txt'
    we = wordEmbedding(fp)
    print(we.generateSimilar("man"))


main()
