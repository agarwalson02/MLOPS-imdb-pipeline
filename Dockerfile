FROM python:3.13.7-slim

WORKDIR /app

COPY flask_app/ .

COPY models/vectorizer.pkl models/vectorizer.pkl

RUN pip install -r requirements.txt

RUN python -m nltk.downloader -d /usr/local/share/nltk_data stopwords wordnet omw-1.4

EXPOSE 5000

CMD ["python","app.py"]

# CMD [ "gunicorn","--bind","0.0.0.0:5000","--timeout","!20","app:app" ]
