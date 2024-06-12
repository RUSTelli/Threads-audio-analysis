from nlp_model import TextClassifier
from data_managing import data_pipeline


dataset = data_pipeline()
bertone = TextClassifier(num_labels=3, output_dir="sentiment_output")


bertone.train(dataset)