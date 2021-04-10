# Personal-Assistant---Python
A personal assistant project called 'Kiya' using Python with a little deep learning involved
Works with Python 3


# Requirements :
-- Python 3
-- pip install numpy
-- pip install tflearn
-- pip install tensorflow
-- pip install json
-- pip install pickle
-- pip install speech_recognition 
-- pip install pyttsx3
-- pip install pywhatkit
-- pip install wikipedia
-- pip install nltk


# Features :
  Speech recognition :
  -- Kiya recognizes your speech and can talk back
  -- Speech recognition runs continuosly in the interval of 5 seconds

  Chat with your assistant :
  -- This project uses 'deep learning' feature for little chats
  -- 'intents.json' file is needed for the deep learning technique
  -- Chats can be triggered by using 'Kiya' in your sentence

  Get info from wikipedia :
  -- If you include 'wikipedia' in you sentence, Kiya would get information from wikipedia
  -- Currently, Kiya gets 2 sentences from wikipedia
  -- Change the number of sentences in --> wiki = wikipedia.summary(kiya_inp, sentences=2)

  Play videos on YouTube :
  -- If you include 'YouTube' in your sentence, Kiya would play videos on YouTube

  Shutdown :
  -- To shutdown Kiya, just say 'Shutdown'
