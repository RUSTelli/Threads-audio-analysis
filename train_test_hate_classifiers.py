from hate_classifier import HateClassifier
from data_managing import data_pipeline

from consts import DATASETS, MODELS

def main():
    """
    Main function to train and test hate type classifiers for different languages.

    This function iterates over the available datasets and performs the following steps:
    1. Prints the language name multiple times.
    2. Prepares the train, eval, and test data using the data_pipeline function.
    3. Initializes the HateClassifier model.
    4. Trains and tests the model using the train_and_test method.
    5. Prints the test metrics for the language.

    Returns:
        None
    """
    for language in DATASETS.keys():
        print(f"{language}\n"*10)

        train_data, eval_data, test_data = data_pipeline(classification_type="hate", language=language, is_multi_lang_model=False)
        model_path = MODELS[language]
        model = HateClassifier(model_path, language, is_multi_lang_model=False)
        _, labels_id, metrics = model.train_and_test(train_data, eval_data, test_data, epochs=4, batch_size=64)
        
        print(f"TEST METRICS - {language}" )
        print(metrics)


if __name__ == "__main__":
    main()