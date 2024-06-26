from speech_recognition import audio_to_text
from sentiment_classifier import SentimentClassifier
from hate_classifier import HateClassifier



# transcribe the audio file using the model
transcribed = audio_to_text()

# load sentiment classifier model
sentiment_classifier = SentimentClassifier.load_model()
# load hate type classifier model

# infer sentiment from the transcribed text
sentiment = sentiment_classifier.predict(transcribed)

# if sentiment is hateful then infer hate type with the hate type classifier model

#hate_type = hate_classifier.predict(transcribed)



