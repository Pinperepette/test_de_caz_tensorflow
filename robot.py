# -*- coding: utf-8 -*-
import pickle, json, nltk, tflearn, random, sys
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tensorflow as tf

def inizializza():
    with open('frasi.json') as json_data:
        intents = json.load(json_data)
    words = []
    classes = []
    documents = []
    ignore_words = ['?']
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            documents.append((w,intent['tag']))
            if(intent['tag'] not in classes):
                classes.append(intent['tag'])
    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))
    training = []
    output = []
    output_empty = [0]*len(classes)
    for doc in documents:
        bag = []
        pattern_words = doc[0]
        patter_words = [stemmer.stem(word.lower()) for word in pattern_words]
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag,output_row])
    random.shuffle(training)
    training = np.array(training)
    train_x = list(training[:,0])
    train_y = list(training[:,1])
    tf.reset_default_graph()
    net = tflearn.input_data(shape=[None,len(train_x[0])])
    net = tflearn.fully_connected(net,8)
    net = tflearn.fully_connected(net,8)
    net = tflearn.fully_connected(net,len(train_y[0]),activation='softmax')
    net = tflearn.regression(net)
    model = tflearn.DNN(net,tensorboard_dir='tflearn_logs')
    model.fit(train_x,train_y,n_epoch=1000,batch_size=8,show_metric=True)
    model.save('model.tflearn')
    pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )

argomenti = len(sys.argv)
if argomenti == 2:
    if sys.argv[1] == "-i":
        print("____INIZIALIZZO____")
        inizializza()
        print("____FATTO_RILANCIA_SCRIPT____")
        sys.exit()
    
data = pickle.load(open("training_data","rb"))
parole = data['words']
classi = data['classes']
train_x = data['train_x']
train_y = data['train_y']

with open('frasi.json') as json_data:
    intents = json.load(json_data)

net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

model.load('./model.tflearn')

def clean_up_sentence(sentence):
    sentence_parole = nltk.word_tokenize(sentence)
    sentence_parole = [stemmer.stem(word.lower()) for word in sentence_parole]
    return sentence_parole


def bow(sentence,parole,show_details=False):
    sentence_parole = clean_up_sentence(sentence)
    bag = [0]*len(parole)
    for s in sentence_parole:
        for i,w in enumerate(parole):
            if(w==s):
                bag[i] = 1
                if(show_details):
                    print("found in bag : %s"%w)
    return(np.array(bag))

context = {}

ERROR_THRESHOLD = 0.25
def classify(sentence):
    results = model.predict([bow(sentence,parole)])[0]
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x:x[1],reverse=True)
    return_list = []
    for r in results:
        return_list.append((classi[r[0]],r[1]))
    return return_list

def response(sentence,userID='123',show_details=False):
    results = classify(sentence)
    if results:
        while results:
            for i in intents['intents']:
                if(i['tag']==results[0][0]):
                    if('context_set' in i):
                        if show_details:
                            print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                    if not 'context_filter' in i or \
                       (userID in context and 'context_filter' in i and i['context_filter']==context[userID]):
                        if show_details:
                            print('Tag : ',i['tag'])
                        return (random.choice(i['responses']))
            results.pop(0)

def main():
    print()
    print("User :",end=" ")
    inp = input()
    while inp!="0":
        bot_reply = response(inp)
        print("Bot : ",bot_reply)
        print("User :",end=" ")
        inp = input()
    print()

if(__name__=="__main__"):
    main()







    
