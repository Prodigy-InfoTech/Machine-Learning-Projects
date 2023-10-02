import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer 
import subprocess

from tensorflow.keras.models import load_model
bot_name = 'Serenity'
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('ANN Chatbot\intents.json').read())

words = pickle.load(open('ANN Chatbot\words.pkl','rb'))
classes = pickle.load(open('ANN Chatbot\classes.pkl','rb'))
model = load_model('ANN Chatbot\chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    bag = [0]* len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD] 

    results.sort(key= lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
        return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result
print("Go! Bot is runnning")

def sere_res(msg):
    res = predict_class(msg)  # Call the predict_class function with the message
    response = get_response(res, intents)  # Pass the prediction result to get_response
    return response

