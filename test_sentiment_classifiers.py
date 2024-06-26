from sentiment_classifier import SentimentClassifier
from data_managing import data_pipeline
from consts import MODELS

language = "it"
model_path = MODELS[language]


trainD, evalD, testD = data_pipeline(classification_type="sentiment", language=language, is_multi_lang_model=False)
lang_specific_model = SentimentClassifier(model_path, language, is_multi_lang_model=False)
output = lang_specific_model.train_and_test(trainD, evalD, testD, epochs=1, batch_size=10)

for element in output:
    print(element)
