import pickle
from random import random
import plotly # pip install plotly
import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA
import wordembedding as we


def display_pca_scatterplot_3D(we, user_input=None, words=None, label=None, color_map=None, topn=5, sample=10):

    if words == None:
        if sample > 0:
            words = np.random.choice(list(we.model.vocab.keys()), sample)
        else:
            words = [ word for word in we.model.vocab ]
    
    word_vectors = np.array([we.model[w] for w in words])
    
    two_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:2]
    gender_subspace = we.findBiasDirections(we.definition_pairs) #x dimension in the gender bias direction
    ## NEED ANALOGIES DIRECTION TO MAKE THIS A 2D ARRAY
    data = []
    count = 0
    
    for i in range (len(user_input)):

                trace = go.Scatter(
                    x = two_dim[count:count+topn,0], 
                    y = two_dim[count:count+topn,1],  
                    text = words[count:count+topn],
                    name = user_input[i],
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 2
                    }
       
                )
                tracez = go.Scatter(
                    x = gender_subspace[count:count+topn,0], 
                    y = gender_subspace[count:count+topn,1],  
                    text = words[count:count+topn],
                    name = user_input[i],
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 2
                    }
       
                )
                
                # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable. Also, instead of using variable three_dim, use the variable that we have declared earlier (e.g two_dim)
            
                data.append(tracez) #gender subspace
                count = count+topn

    trace_input = go.Scatter(
                    x = two_dim[count:,0], 
                    y = two_dim[count:,1],  
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

    # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable.  Also, instead of using variable three_dim, use the variable that we have declared earlier (e.g two_dim)
            
    data.append(trace_input)    
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
    similar_word = [word[0] for word in plot_data]
    similarity = [word[1] for word in plot_data] 
    similar_word.extend(user_input)
    labels = [word[2] for word in plot_data]
    label_dict = dict([(y,x+1) for x,y in enumerate(set(labels))]) #??
    color_map = [label_dict[x] for x in labels]
    return [user_input, similar_word,similarity,labels,label_dict, color_map]

# display_pca_scatterplot_3D(model, user_input, similar_word, labels, color_map)
embedding = we.WordEmbedding('/Users/darrylyork3/Desktop/Comps22/homemakers/data/w2v_gnews_super_small.txt')

sim_words = embedding.generateNSimilar('actress, aunt, leopard, bachelor', 2) # Work on this 

plot_data = format_plot_data(sim_words, 'actress, aunt, leopard, bachelor')

display_pca_scatterplot_3D(embedding, plot_data[0], plot_data[1], plot_data[2], plot_data[3])