FROM python:3.11-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=$OPENAI_API_KEY

VOLUME [ "/app/transcribe-audio" ]

CMD [ "python3", "-u", "transcribe.py" ]

