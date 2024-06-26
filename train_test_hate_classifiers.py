from hate_classifier import HateClassifier
from data_managing import data_pipeline

from consts import DATASETS, MODELS


for language in DATASETS.keys():
    for _ in range(10): 
        print(f"{language}")

    for epoch in [4,5]:
        print(f"Epoch: {epoch}")

        train_data, eval_data, test_data = data_pipeline(classification_type="hate", language=language, is_multi_lang_model=False)
        model_path = MODELS[language]
        lang_specific_model = HateClassifier(model_path, language, is_multi_lang_model=False)
        _, _, metrics = lang_specific_model.train_and_test(train_data, eval_data, test_data, epochs=epoch, batch_size=64)
        print(f"TEST METRICS ------------{language}" )
        print(metrics)