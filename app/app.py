from fastapi import FastAPI
import nltk
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model
from helper_functions import get_pre_processed_input
app = FastAPI()

#defining variables to load word2vec model and deep learning model
sentiment_classifier = None
word2vec_model = None

# Define the event handler for server startup
@app.on_event("startup")
async def startup_event():
    # Check if the required NLTK files are already downloaded
    if not nltk.data.find("tokenizers/punkt"):
        # Download the required NLTK files
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')

    #loading the ml model
    sentiment_classifier = load_model("trained_models/sentiment_model.h5")
    word2vec_model = Word2Vec.load("trained_models/word2vec.model")


#base route
@app.get("/",tags=['ROOT'])
async def root() -> dict:
    """ Base Route for the project """
    return {
        "data": "Welcome to Sentiment Analysis"
    }
