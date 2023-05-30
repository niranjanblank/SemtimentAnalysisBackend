from fastapi import FastAPI

app = FastAPI()

#base route
@app.get("/",tags=['ROOT'])
async def root() -> dict:
    """ Base Route for the project """
    return {
        "data": "Welcome to Sentiment Analysis"
    }
