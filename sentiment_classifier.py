from transformers import AutoModelForSequenceClassification,Trainer, TrainingArguments, AutoTokenizer
from scipy.special import softmax
import numpy as np
import evaluate
import torch
import os

from consts import LABEL2ID_B, ID2LABEL_B
from tokenizer import get_tokenizer

class SentimentClassifier():
    def __init__(self, model_path:str, language:str, is_multi_lang_model=False, save_path:str=None):
        self.tokenizer  = get_tokenizer(language, is_multi_lang_model)
        self.output_dir = os.path.join("sentiment", language, model_path, "checkpoints")
        self.save_path  = save_path
        self.model      = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            id2label=ID2LABEL_B,
            label2id=LABEL2ID_B,
        )


    # def _save_model(self):
    #     self.model.save_pretrained(self.save_path)
    #     self.tokenizer.save_pretrained(self.save_path)


    def train(self, dataset, epochs=2, batch_size=16):
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
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=self.tokenizer,
            compute_metrics=_compute_metrics,
        )

        trainer.train()
        trainer.save_model(self.save_path)


    @classmethod
    def load_model(cls, model_path:str, language:str, is_multi_lang_model=False):
        loaded_model = cls(model_path, language, is_multi_lang_model)
        # loaded_model.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        # loaded_model.tokenizer = get_tokenizer(language)
        return loaded_model


    def predict(self, texts):
        self.model.eval()  # Ensure model is in evaluation mode
        predictions = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=150)
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits
            probs = softmax(logits.numpy(), axis=1)
            prediction = np.argmax(probs, axis=1)
            predictions.append(prediction[0])
        return predictions
    
