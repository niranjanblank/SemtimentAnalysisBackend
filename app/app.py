from fastapi import FastAPI
import nltk
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model
from helper_functions import get_pre_processed_input
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import os

app = FastAPI()

#configuring middleware for cors setting
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with the specific origins you want to allow
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#defining variables to load word2vec model and deep learning model
sentiment_classifier = None
word2vec_model = None

# Define the event handler for server startup
@app.on_event("startup")
async def startup_event():
    global sentiment_classifier
    global word2vec_model

    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Set the TF_ENABLE_ONEDNN_OPTS environment variable
    tf.config.threading.set_intra_op_parallelism_threads(1)  # Limit TensorFlow to use only one thread for parallelism
    # Check if the required NLTK files are already downloaded
    if not nltk.data.find("tokenizers/punkt"):
        # Download the required NLTK files
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')

    #loading the deep learning model and gensim word2vec model
    sentiment_classifier = load_model("trained_models/sentiment_model.h5")
    word2vec_model = Word2Vec.load("trained_models/word2vec.model")


#base route
@app.get("/",tags=['ROOT'])
async def root() -> dict:
    """ Base Route for the project """
    return {
        "data": "Welcome to Sentiment Analysis"
    }

@app.post("/sentiment",tags=['sentiment'])
async def predict_sentiment(text: dict) -> dict:
    """
    Route for sentiment analysis of text
    """
    tweet = text['data']
    #pre process the text to get in format recognizable by the deep learning model
    processed_tweet = get_pre_processed_input(tweet, word2vec_model)
    #predicting
    prediction = sentiment_classifier.predict(processed_tweet)
    prediction_score = float(prediction[0,0])
    #calculating the prediction label
    if prediction_score>0.5:
        predicted_label = 1
        confidence = prediction_score
    else:
        predicted_label = 0
        confidence = 1 - prediction_score
    return {
        "data": tweet,
        "prediction_score": prediction_score,
        "predicted_label": predicted_label,
        "confidence": confidence
    }
