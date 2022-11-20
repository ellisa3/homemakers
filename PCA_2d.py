import pandas as pd
from random import random
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from sklearn.decomposition import PCA
import wordembedding as we
import json

### A visualization of the word embedding projected onto the gender direction
### Missing the linear svm data to project gender neutrality along the y-axis
### Allows for debiased embedding visualization 

def rand_jitter(arr):
    stdev = .01 * (max(arr) - min(arr))
    return arr + (np.random.randn(len(arr)) * stdev)

def display_pca_scatterplot_2D(we, words=None, sample=20):

    if sample < 10:
        words.append(np.random.choice(list(we.model.key_to_index), sample))
    elif words == None:
        words = [ word for word in we.model.key_to_index ]
    
    word_vectors = np.array([we.model[w] for w in words])

    
    gender_subspace = we.doPCA(we.definition_pairs).transform(word_vectors)[:,:2] #x dimension in the gender bias direction (.compintnts)

    data = []
    count = 0

    # plot inputs, edits to shape of markers 
    trace_inputz = go.Scatter(
                    x = rand_jitter(gender_subspace[count:,0]), 
                    y = rand_jitter(gender_subspace[count:,1]),  
                    text = words[count:],
                    textposition = "middle center",
                    textfont_size = 30,
                    mode = 'text',
                    marker = {
                        'size': 1,
                        'opacity': 1,
                        'color': 'silver'
                    },
                    showlegend=False,
                    cliponaxis=True
                    )
    
    he_she = go.Scatter(
                        x = np.array([-.35, .35]),
                        y = np.array([-.004,-.004]),
                        text = ["she", "he"],
                        textposition = "middle center",
                        textfont_size = 40,
                        mode = 'text',
                        marker = {
                            'size': 1,
                            'opacity': 1,
                            'color': 'red'
                        },
                        showlegend= False,
                        cliponaxis=True
                        )

    
            
    data.append(trace_inputz)
    data.append(he_she)   

    layout = go.Layout(
        margin = {'l': 50, 'r': 50, 'b':50, 't': 50},
        showlegend=False,
        font = dict(
            family = " PT Sans Narrow ",
            size = 15),
        autosize = False,
        width = 1800,
        height = 1000,
        plot_bgcolor= 'rgb(255,255,255)',
        xaxis=go.layout.XAxis(showgrid=False, zeroline=True, zerolinecolor='red', zerolinewidth=2, showticklabels=False, showline=False),
        yaxis=go.layout.YAxis(showgrid=False, zeroline=True, zerolinecolor='red', zerolinewidth=2, showticklabels=False, showline=False)
    ) 


    plot_figure = go.Figure(data = data, layout = layout)
    plot_figure.write_image("/content/homemakers/visualizations/DebiasScatterplot.svg", format="svg")
    # plot_figure.show()

input_words=[]
embedding = we.WordEmbedding('/content/homemakers/data/w2v_gnews_small.txt')

numwords=0
for word in embedding.scatterplot:

    if word in embedding.model.key_to_index: 
        input_words.append(word)
        numwords+=1


# Uncomment this for debiased visual
# with open("/content/homemakers/data/genderedPaper.json") as gpfile:
#     gendered_paper = json.load(gpfile)
#     embedding.debias(gendered_paper)

display_pca_scatterplot_2D(embedding, words=input_words)