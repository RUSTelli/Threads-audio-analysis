from nlp_model import HateClassifier
from data_managing import data_pipeline


hate_dataset      = data_pipeline(classification_type="multi")

hate_classifier      = HateClassifier()

hate_classifier.train(hate_dataset)