FROM python:3.11-slim-buster

RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt \
    && python3 -m nltk.downloader punkt stopwords wordnet

COPY . .

ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=$OPENAI_API_KEY

ENTRYPOINT [ "python3", "transcribe.py" ]
CMD ["https://youtu.be/bZe5J8SVCYQ"]
