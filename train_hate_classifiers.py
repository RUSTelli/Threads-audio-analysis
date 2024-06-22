from data_managing import data_pipeline
from hate_classifier import HateClassifier 
from consts import DATASETS, MODELS

#train each language-specific model on sentiment classification
for language in DATASETS.keys():
    dataset = data_pipeline(classification_type="hate", language=language, is_multi_lang_model=False)
    model_path = MODELS[language]
    lang_specific_model = HateClassifier(model_path, language, is_multi_lang_model=False)
    lang_specific_model.train(dataset, epochs=1, batch_size=16)


for language in DATASETS.keys():
    dataset = data_pipeline(classification_type="hate", language=language, is_multi_lang_model=True)
    model_path = MODELS["multi"]
    multi_lang_model = HateClassifier(model_path, language, is_multi_lang_model=True)
    multi_lang_model.train(dataset, epochs=3, batch_size=64)