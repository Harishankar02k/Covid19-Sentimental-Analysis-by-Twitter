import os, sys, re, getopt
import pandas as pd
pd.options.mode.chained_assignment = None 
import numpy as np
import re
import nltk
from gensim.models import word2vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
import csv

if sys.version_info[0] < 3:
    raise Exception("You must use at least Python 3.x")
 
#Plotting function from Jeff Delaney's Kaggle Notebook
# https://www.kaggle.com/jeffd23/visualizing-word-vectors-with-t-sne
def tsne_plot(model, month): 
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    print("=========" + "Plotting the vectors" +"===========")  
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens[:int(len(tokens)/2)])

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.title(month + ' Top Tweets: 2500 Words Vector Map')
    plt.savefig('results/word_vectors_'+month+'.png')
    plt.show()

def create_word2vec_model(filepath, oneplot=False):
    corpus = pd.read_csv(filepath, sep=",")#.sample(5000, random_state=23) 
    corpus = corpus.to_numpy().flatten()  
    voc = [] 
    for i in range(int(len(corpus))): 
        voc.append(corpus[i].split(" "))  
    model = word2vec.Word2Vec(min_count=4, size= 20, workers=6, window =3) 
    model.build_vocab(voc, progress_per=10000) 
    print("===========" + "Training model" +"============")  
    model.train(corpus, total_examples=1, epochs=1) 

    print("=========" + "Training complete" +"===========")  
    return model
# print(model.similarity('coronavirus', 'quarantine')) 

def most_similar_words(model, wordlist):
    for word in wordlist: 
        if word in model.wv:
            print("most similar words to " + word +": ", model.wv.most_similar(word))   
  
def generate_wordvector():
    main("")

def main(arg):
    if len(arg) == 0:
        path = "data/tables/"
        # model = create_word2vec_model(path, oneplot=True) 
        tweet_paths = ["May_top_tweets.csv", "Apr_top_tweets.csv", "Mar_top_tweets.csv", "Feb_top_tweets.csv", "Jan_top_tweets.csv", "Dec_top_tweets.csv"] 
        wordlist = ["corona", "china", "trump", "coronavirus"]
        for t in tweet_paths:
            model = create_word2vec_model(path+t)
            most_similar_words(model, wordlist)
            tsne_plot(model, t[:3]) 
    else:
        try:
            opts, args = getopt.getopt(arg, "", ("data_filepath="))
            for opt, arg in opts:
                if opt == '--data_filepath':
                    path = arg
        except getopt.GetoptError as err:
            print('Arguments parser error, try -h')
            print('\t' + str(err))

        print("Word vectors are being created.")

        tweet_paths = ["May_top_tweets.csv", "Apr_top_tweets.csv", "Mar_top_tweets.csv", "Feb_top_tweets.csv", "Jan_top_tweets.csv", "Dec_top_tweets.csv"][::-1]
        wordlist = ["corona", "china", "trump", "coronavirus"]
        for t in tweet_paths:
            model = create_word2vec_model(path+tweet_paths)
            most_similar_words(model, wordlist)
            tsne_plot(model)

if __name__ == '__main__':
    main(sys.argv[1:])
