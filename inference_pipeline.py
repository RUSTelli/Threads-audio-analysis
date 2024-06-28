from speech_recognition import audio_to_text
from sentiment_classifier import SentimentClassifier
from hate_classifier import HateClassifier
from consts import SENTIMENT_WEIGHTS_PATH, HATE_WEIGHTS_PATH



# transcribe the audio file using the model
transcribed = audio_to_text()

# load sentiment classifier model
sentiment_classifier = SentimentClassifier(SENTIMENT_WEIGHTS_PATH, "it")
# load hate type classifier model
hate_classifier = HateClassifier(HATE_WEIGHTS_PATH, "it")

# infer sentiment from the transcribed text
sentiment = sentiment_classifier.predict(transcribed)

if sentiment == "HATEFUL":
    hate_type = hate_classifier.predict(transcribed)
    print(hate_type)



