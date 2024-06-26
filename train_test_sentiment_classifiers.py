from sentiment_classifier import SentimentClassifier
from data_managing import data_pipeline
from consts import MODELS, DATASETS


for language in DATASETS.keys():
    for _ in range(10): 
        print(f"{language}")

    train_data, eval_data, test_data = data_pipeline(classification_type="sentiment", language=language, is_multi_lang_model=False)
    model_path = MODELS[language]
    lang_specific_model = SentimentClassifier(model_path, language, is_multi_lang_model=False)
    _, _, metrics = lang_specific_model.train_and_test(train_data, eval_data, test_data, epochs=4, batch_size=64)
    print(f"TEST METRICS ------------{language}" )
    print(metrics)
