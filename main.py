import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import speech_recognition as sr 
import pyttsx3
import pywhatkit
import wikipedia
import nltk
import datetime
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

try:
    nltk.download('punkt')
except:
    pass

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"] :
        for pattern in intent["patterns"] :
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
        if intent["tag"] not in labels :
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)
    training = []
    output = []
    out_empty = [ 0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x) :
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words :
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
   
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.compat.v1.reset_default_graph()
net = tflearn.input_data(shape = [None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
#try:
#    model.load("model.tflearn")
#except:
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    
    return numpy.array(bag)

def chat(voice_in):
    while True:
        inp = voice_in
        results = model.predict([bag_of_words(inp,words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        
        if results[results_index] > 0.7 :

            for tg in data["intents"]:
                if tg['tag'] == tag :
                    responses = tg['responses']
            
            return_output = (random.choice(responses))
        else :
            return_output = "Kiya currently does not understand you!"
        
        return return_output

def run_kiya():
    try:
        r = sr.Recognizer()
        engine = pyttsx3.init()
        with sr.Microphone() as source:

            while (True):
                print("Start talking (type quit to stop)")
                r.adjust_for_ambient_noise(source)
                #r.pause_threshold = 1
                audio = r.listen(source,phrase_time_limit=5)
                kiya_inp = r.recognize_google(audio)
                print('Kiya recognizes : ', kiya_inp)
                kiya_inp = kiya_inp.lower()
                if 'shutdown' in kiya_inp:
                    engine.say("Kiya is shutting down")
                    global kiya_quit 
                    kiya_quit = 1
                    break
                elif 'youtube' in kiya_inp:
                    kiya_inp = kiya_inp.replace('youtube', '')
                    pywhatkit.playonyt(kiya_inp)
                elif 'wikipedia' in kiya_inp:
                    kiya_inp = kiya_inp.replace('wikipedia', '')
                    wiki = wikipedia.summary(kiya_inp, sentences=2)
                    engine.say("According to wikipedia")
                    engine.say(wiki)
                elif 'time' in kiya_inp:
                    strTime = datetime.now().strftime("%H:%M:%S")    
                    engine.say("the time is", strTime)
                elif 'kiya' in kiya_inp:
                    kiya_inp = kiya_inp.replace('kiya', '')
                    get_output = chat(kiya_inp)
                    #voices = engine.getProperty('voices')
                    #engine.setProperty('voice', voices[0].id)
                    engine.say(get_output)
                else :
                    print("Kiya did not get that, please use Kiya in your sentence for me to respond")
                    engine.say("Kiya did not get that, please use Kiya in your sentence for me to respond")

                engine.runAndWait()
    except:
        pass


kiya_quit = 0
while kiya_quit == 0:
    run_kiya()











