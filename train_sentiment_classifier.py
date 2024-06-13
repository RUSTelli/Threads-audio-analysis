from nlp_model import SentimentClassifier
from data_managing import data_pipeline


sentiment_dataset   = data_pipeline()
#hate_dataset      = data_pipeline(classification_type="multi")

sentiment_classifier = SentimentClassifier()
# #hate_classifier      = TextClassifier(num_labels=7, output_dir="hate_output")

sentiment_classifier.train(sentiment_dataset, epochs=3, batch_size=128)
# #hate_classifier.train(hate_dataset)