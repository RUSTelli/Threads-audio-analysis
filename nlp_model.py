from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import evaluate
from tokenizer import TOKENIZER
from consts import LABEL2ID, ID2LABEL

class TextClassifier():
    def __init__(self, num_labels=2, output_dir="sentiment_output"):
        self.tokenizer  = TOKENIZER
        self.num_labels = num_labels
        self.output_dir = output_dir
        self.model      = AutoModelForSequenceClassification.from_pretrained(
            "osiria/distilbert-base-italian-cased", 
            config="nlp_config.json",
            num_labels=num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )


    def train(self, dataset, epochs=2):
        def _compute_metrics(eval_pred):
            # TODO manage multi-label metrics
            metrics        = evaluate.combine(["accuracy", "f1", "precision", "recall"])
            logits, labels = eval_pred
            predictions    = np.argmax(logits, axis=-1)
            return metrics.compute(predictions=predictions, references=labels)
    
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=epochs,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=self.tokenizer,
            compute_metrics=_compute_metrics,
        )

        trainer.train()

    def predict(self, test_dataset):
        pass


