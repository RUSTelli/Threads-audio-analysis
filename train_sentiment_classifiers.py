from data_managing import data_pipeline
from nlp_models import SentimentClassifier
from consts import DATASETS, MODELS

# train each language-specific model on sentiment classification
for language in DATASETS.keys():
    dataset = data_pipeline(classification_type="sentiment", language=language, multi_language=False)
    model_path = MODELS[language]
    lang_specific_model = SentimentClassifier(model_path, language, is_multi_lang_model=False)
    lang_specific_model.train(dataset, epochs=3, batch_size=64)


for language in DATASETS.keys():
    dataset = data_pipeline(classification_type="sentiment", language=language, multi_language=True)
    model_path = MODELS["multi"]
    multi_lang_model = SentimentClassifier(model_path, language, is_multi_lang_model=True)
    multi_lang_model.train(dataset, epochs=3, batch_size=64)