import pandas as pd
import os
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import numpy as np
import pickle
import string
import random

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# stop_words = nltk.corpus.stopwords.words('english')

from fastapi import FastAPI
from pydantic import BaseModel

import warnings

warnings.simplefilter('ignore')

app = FastAPI()


class QuestionText(BaseModel):
    context: str


class ResponseText(BaseModel):
    answer: str


@app.get('/')
def index():
    return {'message': 'hello world'}


GREETING_INPUTS = ("hello", "hi", "greetings", "good morning", "good day", "hey", "i need help", "greetings")
GREETING_RESPONSES = ["Good day, How may i of help?", "Hello, How can i help?", "hello",
                      "I am glad! You are talking to me."]


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


lemmer = nltk.WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


# Remove punctuation
def RemovePunction(tokens):
    return [t for t in tokens if t not in string.punctuation]


# Create a stopword list from the standard list of stopwords available in nltk
stop_words = set(stopwords.words('english'))


def Talk_To_ProdBot(test_set_sentence):
    json_file_path = "conversation_json.json"
    tfidf_vectorizer_pickle_path = "tfidf_vectorizer.pkl"
    tfidf_matrix_pickle_path = "tfidf_matrix_train.pkl"

    i = 0
    sentences = []

    # ---------------Tokenisation of user input -----------------------------#

    tokens = RemovePunction(nltk.word_tokenize(test_set_sentence))
    pos_tokens = [word for word, pos in pos_tag(tokens)]

    word_tokens = LemTokens(pos_tokens)

    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    filtered_sentence = " ".join(filtered_sentence).lower()

    test_set = (filtered_sentence, "")

    # For Tracing, comment to remove from print.
    print('FILTERED INPUT->' + filtered_sentence)

    # -----------------------------------------------------------------------#

    try:
        # ---------------Use Pre-Train Model------------------#
        f = open(tfidf_vectorizer_pickle_path, 'rb')
        tfidf_vectorizer = pickle.load(f)
        f.close()

        f = open(tfidf_matrix_pickle_path, 'rb')
        tfidf_matrix_train = pickle.load(f)
        f.close()
        # ---------------------------------------------------#
    except:
        # ---------------To Train------------------#

        with open(json_file_path) as sentences_file:
            reader = json.load(sentences_file)

            # ---------------Tokenisation of training input -----------------------------#

            for row in reader:
                db_tokens = RemovePunction(nltk.word_tokenize(row['Questions']))
                pos_db_tokens = [word for word, pos in pos_tag(db_tokens, tagset='universal')]
                db_word_tokens = LemTokens(pos_db_tokens)

                db_filtered_sentence = []
                for dbw in db_word_tokens:
                    if dbw not in stop_words:
                        db_filtered_sentence.append(dbw)

                db_filtered_sentence = " ".join(db_filtered_sentence).lower()

                # Debugging Checkpoint
                print('TRAINING INPUT: ' + db_filtered_sentence)

                sentences.append(db_filtered_sentence)
                i += 1
                # ---------------------------------------------------------------------------#

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix_train = tfidf_vectorizer.fit_transform(sentences)
        # print(tfidf_matrix_train)

        f = open(tfidf_vectorizer_pickle_path, 'wb')
        pickle.dump(tfidf_vectorizer, f)
        f.close()

        f = open(tfidf_matrix_pickle_path, 'wb')
        pickle.dump(tfidf_matrix_train, f)
        f.close()
        # ------------------------------------------#

    # use the learnt dimension space to run TF-IDF on the query
    tfidf_matrix_test = tfidf_vectorizer.transform(test_set)
    # print(tfidf_matrix_test)

    # then run cosine similarity between the 2 tf-idfs
    cosine = cosine_similarity(tfidf_matrix_test, tfidf_matrix_train)

    # print only observation and remove it
    # print(cosine.argsort())

    # if not in the topic trained.no similarity
    idx = cosine.argsort()[0][-2]
    flat = cosine.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if req_tfidf == 0:  # Threshold A

        cosine = np.delete(cosine, 0)
        not_understood = "Apology, I do not understand. Can you repeat it one more time elaborately?"

        return not_understood

    else:

        # print(cosine)
        cosine = np.delete(cosine, 0)
        # print(cosine)

        # get the max score
        max = cosine.max()
        response_index = 0

        # if max score is lower than < 0.5 > (we see can ask to rephrase.)
        if (max <= 0.5):  # Threshold B

            not_understood = "Apology, I do not understand. Can you rephrase?"

            return not_understood
        else:

            # else we would simply return the highest score
            response_index = np.where(cosine == max)[0][0] + 2

            j = 0

            with open(json_file_path, "r") as sentences_file:
                reader = json.load(sentences_file)
                for row in reader:
                    j += 1
                    if j == response_index:
                        return row["Answers"]
                        break


# print(Talk_To_ProdBot("What was lowest package in production engineering department of this year?"))
#
@app.post("/chat", response_model=ResponseText)
def getquestion(request: QuestionText):
    # try:
    context = request.context
    ques = Talk_To_ProdBot(context)
    return ResponseText(answer =  ques)
