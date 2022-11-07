import pickle
from random import random
import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA
import wordembedding as we
import json


def debiased(embedding, words):
    with open("./data/genderedPaper.json") as gpfile:
        gendered_paper = json.load(gpfile)
    embedding.debias(gendered_paper)

    word_vectors = np.array([we.model[w] for w in words])

    gender_subspace = we.doPCA(we.definition_pairs).transform(word_vectors)[:,:2] #x dimension in the gender bias direction (.compintnts)

    # print(gender_subspace)
    data = []
    count = 0

    # plot inputs, edits to shape of markers 
    trace_inputz = go.Scatter(
                    x = rand_jitter(gender_subspace[count:,0]), 
                    y = rand_jitter(gender_subspace[count:,1]),  
                    text = words[count:],
                    # name = 'input words',
                    textposition = "middle center",
                    textfont_size = 20,
                    mode = 'text',
                    marker = {
                        'size': 1,
                        'opacity': 1,
                        'color': 'black'
                    },
                    showlegend=False,
                    cliponaxis=True
                    )
    
    she_he = go.Scatter(
                        x = np.array([-.5, .5]),
                        y = np.array([-.004,-.004]),
                        text = ["she", "he"],
                        textposition = "middle center",
                        textfont_size = 20,
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
    data.append(she_he)    
    # Configure the layout

    layout = go.Layout(
        margin = {'l': 50, 'r': 50, 'b':50, 't': 50},
        showlegend=False,
        # legend=dict(x=1,y=0.5,font=dict(family="Courier New",size=25,color="black")),
        font = dict(
            color = "black",
            family = " PT Sans Narrow ",
            size = 15),
        autosize = False,
        width = 2000,
        height = 1000,
        plot_bgcolor= 'rgb(255,255,255)',
        xaxis=go.layout.XAxis(showgrid=False, zeroline=True, zerolinecolor='red', zerolinewidth=2, showticklabels=False, showline=False, griddash="longdash"),
        yaxis=go.layout.YAxis(showgrid=False, zeroline=True, zerolinecolor='red', zerolinewidth=2, showticklabels=False, showline=False)
    ) 


    plot_figure = go.Figure(data = data, layout = layout)
    plot_figure.write_image("/Users/darrylyork3/Desktop/Comps22/homemakers/visualizations/GenderDirectionScatterplot.svg", format="svg")
    plot_figure.show()


def rand_jitter(arr):
    stdev = .01 * (max(arr) - min(arr))
    return arr + (np.random.randn(len(arr)) * stdev)

def display_pca_scatterplot_2D(we, words=None, sample=20):

    if sample < 10:
        words.append(np.random.choice(list(we.model.key_to_index), sample))
    elif words == None:
        words = [ word for word in we.model.key_to_index ]
    
    # print(words)
    word_vectors = np.array([we.model[w] for w in words])

    
    gender_subspace = we.doPCA(we.definition_pairs).transform(word_vectors)[:,:2] #x dimension in the gender bias direction (.compintnts)

    # print(gender_subspace)
    data = []
    count = 0

    # plot inputs, edits to shape of markers 
    trace_inputz = go.Scatter(
                    x = rand_jitter(gender_subspace[count:,0]), 
                    y = rand_jitter(gender_subspace[count:,1]),  
                    text = words[count:],
                    # name = 'input words',
                    textposition = "middle center",
                    textfont_size = 30,
                    mode = 'text',
                    marker = {
                        'size': 1,
                        'opacity': 1,
                        'color': 'black'
                    },
                    showlegend=False,
                    cliponaxis=True
                    )
    
    he_she = go.Scatter(
                        x = np.array([-.35, .35]),
                        y = np.array([-.004,-.004]),
                        text = ["he", "she"],
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
    # Configure the layout

    layout = go.Layout(
        margin = {'l': 50, 'r': 50, 'b':50, 't': 50},
        showlegend=False,
        # legend=dict(x=1,y=0.5,font=dict(family="Courier New",size=25,color="black")),
        font = dict(
            color = "black",
            family = " PT Sans Narrow ",
            size = 15),
        autosize = False,
        width = 1800,
        height = 1000,
        plot_bgcolor= 'rgb(255,255,255)',
        xaxis=go.layout.XAxis(showgrid=False, zeroline=True, zerolinecolor='red', zerolinewidth=2, showticklabels=False, showline=False, griddash="longdash"),
        yaxis=go.layout.YAxis(showgrid=False, zeroline=True, zerolinecolor='red', zerolinewidth=2, showticklabels=False, showline=False)
    ) 


    plot_figure = go.Figure(data = data, layout = layout)
    # plot_figure.write_image("/Users/darrylyork3/Desktop/Comps22/homemakers/visualizations/GenderDirectionScatterplot.svg", format="svg")
    plot_figure.show()


input_words=[]
embedding = we.WordEmbedding('/Users/darrylyork3/Desktop/Comps22/homemakers/data/w2v_gnews_small.txt')

numwords=0
for word in embedding.scatterplot:
    print(word)
    if word in embedding.model.key_to_index: 
        input_words.append(word)
        numwords+=1
print(numwords)
display_pca_scatterplot_2D(embedding, words=input_words)
# debiased(embedding,input_words)