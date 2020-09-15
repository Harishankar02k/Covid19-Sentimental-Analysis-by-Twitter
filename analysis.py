import os, sys, re, getopt
import pandas as pd
import csv
import re  
from textblob import TextBlob 
import preprocessor as p
import numpy as np
import emoji
import nltk 
import matplotlib.pyplot as plt 
import matplotlib.font_manager as font_manager
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
import pylab
from wordvector import *
import importlib

if sys.version_info[0] < 3:
    raise Exception("You must use at least Python 3.x")

#Tweet cleaning function from Dilan Jayasekara's Telemedicine Twitter repo
#https://www.dropbox.com/s/5enwrz2ggswns56/Telemedicine_twitter_v3.5.py?dl=0
def clean_tweets(tweet): 
    emoticons_happy = set([
        ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
        ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
        '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
        'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
        '<3'
        ])

        # Sad Emoticons
    emoticons_sad = set([
        ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
        ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
        ':c', ':{', '>:\\', ';('
        ])

        #Emoji patterns
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)

    #combine sad and happy emoticons
    emoticons = emoticons_happy.union(emoticons_sad) 
    tweet = re.sub(r':', '', tweet)
    tweet = re.sub(r'‚Ä¶', '', tweet)
    chars = [';', '``', '"', '.', '//', '\'', '...', ':', ',']
    tweet = emoji.demojize(tweet) 
    for c in chars:
        if c in tweet:
            tweet.replace(c, "") 
    tweet = ' '.join(tweet.split())

    #replace consecutive non-ASCII characters with a space
    tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)
    tweet = emoji_pattern.sub(r'', tweet)

    words = nltk.word_tokenize(tweet)
    tweet= [word for word in words if word.isalnum()]
    tweet= " ".join(tweet)  
    stopwords = nltk.corpus.stopwords.words('english')
    word_tokens = word_tokenize(tweet) 
    content = [w for w in word_tokens if w.lower() not in stopwords] 
    return " ".join(content)

def tweet_processor(path, part, freq=1):  
    myFile = pd.read_csv(path, sep=',')
    tweets = myFile["text"]
    if "May" in path: 
        part = 1
    # if "May" not in path:
    tweets = tweets[int(len(tweets)*(part-1)*0.5):int(len(tweets)*part*0.5)] 
    blob =  " ".join(myFile["text"]).split(" ") 
    processed_tweets = []
    compound_sent = [] 
    print("n tweets: ",len(tweets))
    sid = SentimentIntensityAnalyzer()
    for tweet in tweets:
        cleaned_tweet = p.clean(tweet.lower())
        filtered_tweet= clean_tweets(cleaned_tweet) 
        ss = sid.polarity_scores(filtered_tweet) 
        cur_sent = [ss['neg'],ss['pos'], ss['neu'], ss['compound']]  
        blob = TextBlob(filtered_tweet)
        Sentiment = blob.sentiment     
        polarity = Sentiment.polarity
        subjectivity = Sentiment.subjectivity
        if filtered_tweet != "" and len(filtered_tweet) >2: 
            processed_tweets.append(filtered_tweet)  
            compound_sent.append(cur_sent)
    # np.savetxt("processed_tweets.csv", processed_tweets, delimiter=",", fmt='%s') 
    compound_sent = np.asarray(compound_sent)
    freqs = []
    
    print("number of words: ",len((" ".join(processed_tweets).split(" ")))) 
    print("unique words: ",len(set(" ".join(processed_tweets).split(" ")))) 

    if freq ==0: #Use blob counting
        words = set(blob.split(" "))
        for word in set(blob.split(" ")):
            if word != "" and len(word)>2: 
                freqs.append([word,blob.count(word)])    
        freqs = np.asarray(freqs)
        freqs = freqs[np.argsort(freqs[:, 1])][::-1]

    if freq ==1: #Use NLTK freqdist
        freqs = pfreq_dist(" ".join(processed_tweets).split(" "))
        freqs = np.asarray(freqs)  
    return processed_tweets, freqs, compound_sent

def make_delta_table(sorted_array, words, month_freqs):
    delta_table = np.zeros((len(words),len(month_freqs)+1), dtype=object)   
    delta_max = np.zeros((len(words), 2), dtype=object)   
    delta_table[:,0] = words 
    delta_max[:,0] = words 
    delta_array = sorted_array[:,1:].astype(np.int)
    dif =  delta_array[:,2:] - delta_array[:,1:-1] 
    delta_table[:,1:-2] = dif 
    delta_max[:,1] = np.max(delta_array,axis = 1) - np.min(delta_array,axis = 1)
    sorted_delta_max = delta_max[np.argsort(-1*np.asarray(delta_max[:, -1],dtype=np.int))]
    np.savetxt("newdelta_table2.csv", delta_table, delimiter=",", fmt='%s')
    np.savetxt("newsorted_delta2_max.csv", sorted_delta_max, delimiter=",", fmt='%s')

#Use this function to process raw Twitter data and create csv tables of all_tweets, 
#corpus, average sentiments per month, frequencies 
def create_freqtable(renew_data=False): 
    paths = ["May_top.csv", "Apr_top.csv", "Mar_top.csv", "Feb_top.csv", "Jan_top.csv", "Dec_top.csv"][::-1]
    mean_comp = [ ]
    words = [] 
    month_freqs = []
    avg_sents = []
    all_sents = []
    all_tweets = []
    for month in paths: 
        print("=======Processing " + month[:-4] + " Tweets=======")
        reps = 3
        if "May" in month: 
            reps = 2
        for i in range(1,reps): 
            tweets, freqs, compound_sent = tweet_processor("data/Top5000tweets/" + month, i) 
            avg_sents.append(np.average(compound_sent, axis=0)) #append average monthly sentiment
            all_sents.append(compound_sent)
            words.extend(freqs[:,0])
            words = list(set(words)) #CORPUS  
            month_freqs.append(freqs) 
            all_tweets += tweets
        # else: 
        #     tweets, freqs, compound_sent = tweet_processor("data/Top5000tweets/" + month, 1) 
        #     avg_sents.append(np.average(compound_sent, axis=0)) #append average monthly sentiment
        #     all_sents.append(compound_sent)
        #     words.extend(freqs[:,0])
        #     words = list(set(words)) #CORPUS  
        #     month_freqs.append(freqs) 
        #     all_tweets.append(tweets) 
        if renew_data:
            #========MONTHLY DATA========#    
            np.savetxt("data/tables/" + month[:-4] + "_tweets.csv", tweets, delimiter=",", fmt='%s')
            np.savetxt("data/tables/" + month[:-4] + "_sentiment.csv", avg_sents, delimiter=",", fmt='%s')
            np.savetxt("data/tables/" + month[:-4] + "_freqs.csv", freqs, delimiter=",", fmt='%s') 
    if renew_data:
        np.savetxt("data/tables/all_tweets.csv", np.asarray(all_tweets).flatten(), delimiter=",", fmt='%s')  
    print("-----TEXT PROCESSING COMPLETE-----") 
    words = np.asarray(words) 
     
    freq_table = np.zeros((len(words),len(month_freqs)+1), dtype=object)   
    freq_table[:,0] = words 
    for i in range(1,len(month_freqs)+1):
        f = month_freqs[i-1] 
        for word, freqq in f:
            freq_table[np.argwhere(freq_table[:,0]==word), i] = freqq 
    sorted_array = freq_table[np.argsort(-1*np.asarray(freq_table[:, -1],dtype=np.int))]
     
    # #========RENEW DATA========#
    if renew_data:
        np.savetxt("data/tables/corpus.csv", words, delimiter=",", fmt='%s')
        make_delta_table(sorted_array[:1000], words[:1000], month_freqs)  
        np.savetxt("data/tables/all_sents.csv", all_sents, delimiter=",", fmt='%s') 
        np.savetxt("data/tables/freqtablesorted.csv", sorted_array, delimiter=",", fmt='%s')
        np.savetxt("results/sorted_word_frequency_table.csv", sorted_array, delimiter=",", fmt='%s')
        np.savetxt("data/tables/avg_sents.csv", avg_sents, delimiter=",", fmt='%s')
        print("-----TABLES SAVED IN 'table/data'----") 

def csv_to_corpus_text(filepath, text= False, token_check= False):
    data = pd.read_csv(filepath, sep=',', header=None).reset_index() 
    data = data.to_numpy()[:,1:].flatten() 
    corpus = [] 
    for i in range(len(data)): 
        corpus.append(data[i])#.split(" "))
    tokens =  " ".join(np.asarray(corpus)).split(" ")
    if token_check:  
        return tokens
    if text:   
        return nltk.Text(tokens)
    return corpus

def corpus_to_tokens(corpus):
    corpus = np.asarray(corpus)
    return " ".join(corpus).split(" ")

def pfreq_dist(text, process = False): 
    fdist1 = nltk.FreqDist(text)      
    common = fdist1.most_common(len(text)) 
    return np.asarray(common) 

def custom_dispersion_plot(text, words, ignore_case=False, title="Lexical Dispersion Plot"):
    """
    Generate a lexical dispersion plot.

    :param text: The source text
    :type text: list(str) or enum(str)
    :param words: The target words
    :type words: list of str
    :param ignore_case: flag to set if case should be ignored when searching text
    :type ignore_case: bool
    """

    try:
        from matplotlib import pylab
    except ImportError:
        raise ValueError(
            "The plot function requires matplotlib to be installed."
            "See http://matplotlib.org/"
        )

    text = list(text)
    words.reverse()

    if ignore_case:
        words_to_comp = list(map(str.lower, words))
        text_to_comp = list(map(str.lower, text))
    else:
        words_to_comp = words
        text_to_comp = text

    points = [
        (x, y)
        for x in range(len(text_to_comp))
        for y in range(len(words_to_comp))
        if text_to_comp[x] == words_to_comp[y]
    ]
    if points:
        x, y = list(zip(*points))
    else:
        x = y = ()
    pylab.plot(x, y, "b|", scalex=0.1)
    pylab.yticks(list(range(len(words))), words, color="b")
    pylab.ylim(-1, len(words))
    pylab.title(title)
    pylab.xlabel("Word Offset")
    pylab.savefig("results/dispersion_plot.png")
    pylab.show()

def lexical_dispersion():
    text1 =  csv_to_corpus_text("data/tables/all_tweets.csv", text =True)
    # tokens = corpus_to_tokens(corpus)
    # text = nltk.Text(tokens)
    # text.collocations()
    # print(text.concordance("coronavirus")) 
    commons = list(pfreq_dist(text1)[0:30,0]) 
    # commons.remove("one")
    # commons.remove("like")
    # commons.remove("get")
    # commons.remove("new")
    # commons.remove("says")
    # commons.remove("may")
    custom_dispersion_plot(text1, list(commons)) 
    custom_dispersion_plot(text1, ["russia", "china","turkey", "united states", "italy", "spain", "uk", "usa", "france"])
    custom_dispersion_plot(text1, ["trump", "pandemic", "lockdown", "spain", "mask"])  
    np.savetxt("results/extracted_features.csv", commons, delimiter=",", fmt='%s') 

def freq_dist(filepath, dist = False): 
    text =  csv_to_corpus_text(filepath, text =True) 
    fdist1 = nltk.FreqDist(text) 
    common = fdist1.most_common(len(text))  
    if dist:
        return fdist1
    return np.asarray(common) 

def lexical_diversity(filepath):
    text =  csv_to_corpus_text("data/tables/all_tweets.csv", text =True) 
    return len(set(text)) / len(text)

def plot_sentiments():
    sents = pd.read_csv("new_avg_sents.csv", sep=',', header=None).reset_index()
    sents.columns = ['months','Negative', 'Positive','Neutral', 'Compound']
    sents['months'] =  ['Dec 15',  'Jan 1', 'Jan 15',  'Feb 1', 'Fab 15', 'Mar 1', 'Mar 15', 'Apr 1', 'Apr 15', ' May1 ', 'May 15']
    sents = sents.set_index('months')
    sents.plot() 

    plt.ylabel("Sentiment")
    plt.xlabel("Date")
    plt.title('Change in average sentiment of Tweets over time')
    plt.rcParams["font.family"] = "Times New Roman"

    plt.savefig('results/average_sentiment_graph.png')
    plt.show() 
    

def freq_graph():
    df = pd.read_csv("data/tables/freqtablesorted.csv", sep=',', header=None)#.reset_index()  
    
    df.columns = ['Words', 'Dec 15',  'Jan1', 'Jan 15',  'Feb 1', 'Feb 15', 'Mar 1', 'Mar 15', 'Apr 1', 'Apr 15', ' May1 ', 'May 15']
    df = df.assign(sums = np.sum(df.T[1:].T, axis=1))  #Add a sum column
    df = df.set_index('Words')
    df = df.sort_values('sums', ascending=False)  
    df.head() 
     
    title_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal',
                'verticalalignment':'bottom'}  
    axis_font = {'fontname':'Arial', 'size':'14'}

    common = freq_dist("all_tweets.csv") 
    common = list(common[0:21,0])  
    common = np.asarray(common)
    df = df.T
    df[common][:-1].plot.area(grid =True)
    plt.style.use('seaborn-whitegrid')

    plt.ylabel("Frequency")
    plt.xlabel("Date")
    plt.title('Most Frequent 20 Words')
    plt.rcParams["font.family"] = "Times New Roman"

    plt.xticks(np.arange(11) , [ 'Dec 15',  'Jan1', 'Jan 15',  'Feb 1', 'Feb 15', 'Mar 1', 'Mar 15', 'Apr 1', 'Apr 15', ' May1 ', 'May 15'])
    # print(df[common])
    # print(common.shape)
    # ax.scatter(df[common], common )
    # ax.set_xlabel("Frequency", fontsize=15)
    # ax.set_ylabel("Date", fontsize=15)
    # ax.set_title('Most Frequent 20 Words')
    # ax.grid(True)
    # fig.tight_layout()

    # plt.show()
    # df[:-1].plot(y=["corona", "china", "war", "pandemic", "rona", "italy", "spain"])
    # df[:-1].plot(y=common[1:20,0], grid = True)
    # df[:-1].plot(y=[ "people", "death", "cdc", "home"])
    plt.savefig('results/frequency_dist_graph.png')
    plt.show()
    

def slang_analysis():
    # df = pd.read_csv("SlangSD.txt", sep='\t', header=None)#.reset_index() 
    slang_corpus = []
    with open("data/SlangSD.txt", newline = "") as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            slang_corpus.append(line) 
    df = pd.read_csv("freqtablesortedlong_50003wsum.csv", sep=',', header=None)#.reset_index()
    df.columns = ['word', 'Dec 15',  'Jan1', 'Jan 15',  'Feb 1', 'Fab 15', 'Mar 1', 'Mar 15', 'Apr 1', 'Apr 15', ' May1 ', 'May 15', 'Sum']
    df = df.sort_values('Sum', ascending=False) 
    
    slang_corpus = np.asarray(slang_corpus)[:,0]
    intersection = np.intersect1d(np.asarray(df['word'], dtype=str), slang_corpus)
    df = df.set_index('word')
    df = df.T
    df[:-1].plot(y=intersection[100:110], grid = True)
    plt.show()
     
def extract_features():
    filepath = "data/tables/all_tweets.csv"
    tweet_words =  csv_to_corpus_text(filepath, text =True) 
    # word_features = freq_dist(filepath, dist =True).keys() 
    word_features = list(freq_dist(filepath, dist =True)) 
    features={} 
    for word in np.asarray(word_features)[:10]:
        print(word)
        # features['contains(%s)' % word]=(word in tweet_words)®
    np.savetxt("results/extracted_features.csv", word_features, delimiter=",", fmt='%s') 
    return features 

def main(arg): 
    mode = 0
    if len(arg) == 0:
        mode = 2 
        if not os.path.exists('data/tables/all_tweets.csv'):
            mode = 1
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('data'):
        print("You have to download data")
        os.makedirs('data')
    if not os.path.exists('data/tables'):
        os.makedirs('data/tables')

    # if len(arg) == 0: 
    #     # print("First run with --process_data and then use --analyze flag")
    try:
        opts, args = getopt.getopt(arg, "", ("data_filepath=", "process_data", "analyze")) 
        for opt, arg in opts:
            if opt == '--process_data': 
                mode = 1 
            if opt == '--analyze': 
                mode = 2

        if mode == 1:
            create_freqtable(renew_data=True)
        if mode == 2 or mode == 1:
            print("Lexical diversity is: ", lexical_diversity(""))
            print("Creating lexical dispersion plot")
            lexical_dispersion()
            print("Creating average sentiment graph")
            plot_sentiments()
            print("Creating frequency distribution graph")
            freq_graph()
            print("Creating word features")
            print("Top 10 word features are:") 
            extract_features()
            print("Geneating word vectors, please be patient...")
            generate_wordvector()
            print("DONE!")
            print("All extracted features and graphs are saved in results.")
            

    except getopt.GetoptError as err:
        print('Arguments parser error, try -h')
        print('\t' + str(err))

if __name__ == '__main__':
    main(sys.argv[1:])
