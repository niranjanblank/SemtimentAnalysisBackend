FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

# setting the work directory
WORKDIR /app

COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download necessary NLTK data
RUN python -m nltk.downloader punkt stopwords wordnet

# Make port 80 available to the world outside this container
EXPOSE 80

# Run main.py when the container launches
CMD ["python","main.py"]