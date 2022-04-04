FROM continuumio/miniconda3:latest

# RUN apt-get update 

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt --user --no-cache-dir