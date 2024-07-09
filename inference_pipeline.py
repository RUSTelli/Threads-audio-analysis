from speech_recognition import audio_to_text
from sentiment_classifier import SentimentClassifier
from hate_classifier import HateClassifier
from consts import SENTIMENT_WEIGHTS_PATH, HATE_WEIGHTS_PATH

def inference_pipeline():
    """
    This function performs an inference pipeline for audio analysis.
    It transcribes an audio file using a model, predicts the sentiment of the transcribed text,
    and if the sentiment is "HATEFUL", it predicts the type of hate.

    NOTE:    Specify the models weight paths in the consts.py file.

    Returns:
        None
    """
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



