from nlp_model import TextClassifier
from data_managing import data_pipeline


sentiment_dataset = data_pipeline(classification_type="ternary")
hate_dataset      = data_pipeline(classification_type="multi")

sentiment_classifier = TextClassifier(num_labels=3, output_dir="sentiment_output")
hate_classifier      = TextClassifier(num_labels=7, output_dir="hate_output")


sentiment_classifier.train(sentiment_dataset)
hate_classifier.train(hate_dataset)