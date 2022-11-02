import pickle
from random import random
import plotly # pip install plotly
import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA
import wordembedding as we


def display_pca_scatterplot_2D(we, user_input=None, words=None, label=None, color_map=None, topn=5, sample=10):

    if words == None:
        if sample > 0:
            words = np.random.choice(list(we.model.vocab.keys()), sample)
        else:
            words = [ word for word in we.model.vocab ]
    
    # print(words)
    word_vectors = np.array([we.model[w] for w in words])

    
    two_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:2]
    gender_subspace = we.doPCA(we.definition_pairs).transform(word_vectors)[:,:2] #x dimension in the gender bias direction (.compintnts)

    # print(gender_subspace)
    
    ## (Axis orthogonal to gender direction)??
    data = []
    count = 0
    
    # For n closest words
    # for i in range (len(user_input)):
    #     trace = go.Scatter(  
    #         x = two_dim[count:count+topn,0], 
    #         y = two_dim[count:count+topn,1],  
    #         text = words[count:count+topn],
    #         name = user_input[i],
    #         textposition = "top center",
    #         textfont_size = 20,
    #         mode = 'markers+text',
    #         marker = {
    #             'size': 10,
    #             'opacity': 0.8,
    #             'color': 2
    #         }

    #     )
    #     tracez = go.Scatter(
    #         x = gender_subspace[count:count+topn,0], 
    #         y = gender_subspace[count:count+topn,1],  
    #         text = words[count:count+topn],
    #         name = user_input[i],
    #         textposition = "top center",
    #         textfont_size = 20,
    #         mode = 'markers+text',
    #         marker = {
    #             'size': 10,
    #             'opacity': 0.8,
    #             'color': 2
    #         }

    #     )
            
    #     data.append(tracez) # topn closest words to input
    #     count = count+topn

    trace_input = go.Scatter(
                    x = two_dim[count:,0], #list of x coord
                    y = two_dim[count:,1],  # list of y coords
                    text = words[count:],   # list of words seperated
                    name = 'input words',   
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 1,
                        'color': 'black'
                    }
                    )

    # plot inputs
    trace_inputz = go.Scatter(
                    x = gender_subspace[count:,0], 
                    y = gender_subspace[count:,1],  
                    text = words[count:],
                    name = 'input words',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 1,
                        'color': 'black'
                    }
                    )

            
    data.append(trace_inputz)    
    # Configure the layout

    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
        x=1,
        y=0.5,
        font=dict(
            family="Courier New",
            size=25,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 15),
        autosize = False,
        width = 1000,
        height = 1000
        )


    plot_figure = go.Figure(data = data, layout = layout)
    plot_figure.show()
    
def format_plot_data(plot_data, input_words):
    user_input = [x.strip() for x in input_words.split(',')]
    # labels = [word[2] for word in plot_data]
    # label_dict = dict([(y,x+1) for x,y in enumerate(set(labels))]) #??
    # color_map = [label_dict[x] for x in labels]
    return [user_input]

# display_pca_scatterplot_3D(model, user_input, similar_word, labels, color_map)
embedding = we.WordEmbedding('/Users/darrylyork3/Desktop/Comps22/homemakers/data/w2v_gnews_small.txt')

input_words = ['actress', 'aunt', 'leopard', 'bachelor', 'in', 'for', 'in']

for pair in embedding.definition_pairs:
    print(pair, pair[0])
    if pair[0] in embedding.model.key_to_index:
        input_words.append(pair[0])
    if pair[1] in embedding.model.key_to_index: 
        input_words.append(pair[1])

display_pca_scatterplot_2D(embedding, words=input_words)
