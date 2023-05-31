import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import numpy as np

def pre_process_tweet(tweet):
    """
    Cleaning the text
    """
    #converting to lowercase
    tweet = tweet.lower()

    #removing hyperlinks
    tweet = re.sub(r'http\S+', '', tweet)

    #removing username from text
    tweet = re.sub(r'@\w+','', tweet)

    #removing hashtags from text
    tweet = re.sub(r'#','',tweet)

    #removing digits
    tweet = re.sub(r'\d+', '', tweet)

    #removing punctuations
    tweet = re.sub(r'[^\w\s]|_', '', tweet)

    # converting to tokens
    tweet_tokens = word_tokenize(tweet)

    # removing stop words
    stop_words = set(stopwords.words('english'))
    cleaned_tweet_tokens = [word for word in tweet_tokens if word not in stop_words ]



    return cleaned_tweet_tokens

def word_to_vec(sentence, word2vec_model):
    """
    Convert a sentence to word embeddings
    """
    return [word2vec_model.wv[word] if word in word2vec_model.wv else np.zeros(word2vec_model.vector_size) for word in sentence]

def get_pre_processed_input(tweet,word2vec_model,maxlen=37,embedding_dim=100):
    """
    function to pre_process the input tweet that includes converting to embedding and getting data in required format by nn model
    """
    # maxlen is 37
    # embedding_dim is 100 which is the vector size of word2vec model
    # Tokenize and pad the tweet
    pre_processed_new_tweet = pre_process_tweet(tweet)
    #getting embedding of each word of tweet
    vectorized_new_tweet =  word_to_vec(pre_processed_new_tweet,word2vec_model)
    # padding to get uniform size inout
    vectorized_new_tweet_padded = pad_sequence(vectorized_new_tweet,word2vec_model, maxlen=maxlen)
    # reshaping the tweet to match input shape of model
    vectorized_new_tweet_padded = vectorized_new_tweet_padded.reshape(1, maxlen, embedding_dim)

    return vectorized_new_tweet_padded


# used to have uniform input by padding the input which are smaller
def pad_sequence(seq, word2vec_model, maxlen=37):
    """
    Function to convert input into uniform size by padding the input which are smaller than maxlen
    """
    return np.array(seq + [np.zeros(word2vec_model.vector_size)] * (maxlen - len(seq)))

