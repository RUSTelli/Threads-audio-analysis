from data_managing import data_pipeline
from nlp_models import SentimentClassifier
from consts import DATASETS, MODELS

for language in DATASETS.keys():
    dataset = data_pipeline(classification_type="sentiment", language=language)
    model = SentimentClassifier(MODELS[language])
    model.train(dataset, epochs=4, batch_size=64)