#generating anologies
#a:x as b:y
# seed pair = [she, he] #[a,b]
# x = [] #literally every word in embedding?
# y = [] #only those who pass the threshold

#pseudocode

# find cosine similarity between she, he
# she_he_similarity = most_similar(positive = [she, topn=10) //solid guess that given she, he will be within top 10 most similar words
# for every word in x
    # cosine_similarities = []
    # cosine_similarities = most_similar(positive = [x], topn=None) //returns all the words in the vocab & their cosine similarity relative to x

    # curr_difference = 0
    # best_difference = 10
    # for theta in cosine_similarities
        # curr_difference = abs_val(she_he_similarity - theta)
        # if curr_difference < best_difference
            # best_difference = curr_difference

    #return best_difference
        
