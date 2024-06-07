from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import evaluate

class Model():
    def __init__(self, num_labels=2):
        self.num_labels = num_labels
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "osiria/distilbert-base-italian-cased", 
            config="custom_nlp_config.json",
            num_labels=num_labels
        )

    def __compute_metrics(eval_pred, ):
        metric = evaluate.load_metric("accuracy")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    def train(self, train_dataset, val_dataset):
        training_args = TrainingArguments(
            output_dir="test_trainer", 
            eval_strategy="epoch"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,                  
            train_dataset=train_dataset,         
            eval_dataset=val_dataset,
            compute_metrics=self.__compute_metrics,
        )

        trainer.train()
