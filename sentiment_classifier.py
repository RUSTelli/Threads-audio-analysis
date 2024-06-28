from transformers import AutoModelForSequenceClassification,Trainer, TrainingArguments
from torch import nn, argmax
import numpy as np
import evaluate
import os

from consts import LABEL2ID_B, ID2LABEL_B, MAX_SEQ_LEN
from tokenizer import get_tokenizer

class SentimentClassifier():
    def __init__(self, model_path:str, language:str, is_multi_lang_model=False):
        self.model      = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            id2label=ID2LABEL_B,
            label2id=LABEL2ID_B,
        )
        self.tokenizer  = get_tokenizer(language, is_multi_lang_model)
        lang            = "multi" if is_multi_lang_model else language
        self.output_dir = os.path.join("sentiment", lang)


    def train_and_test(self, train_dataset, eval_dataset, test_dataset, epochs=2, batch_size=16):
        def _compute_metrics(eval_pred):
            metrics        = evaluate.combine(["accuracy", "f1", "precision", "recall"])
            logits, labels = eval_pred
            predictions    = np.argmax(logits, axis=-1)
            return metrics.compute(predictions=predictions, references=labels)
    
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
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
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=_compute_metrics,
        )

        trainer.train()
        return trainer.predict(test_dataset)
    

    def predict(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")
        outputs = self.model(**inputs)
        probs = nn.functional.softmax(outputs.logits, dim=-1)
        y = argmax(probs, dim=1)
        return ID2LABEL_B[y.item()]