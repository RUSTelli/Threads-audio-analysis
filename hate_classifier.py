from transformers import AutoModelForSequenceClassification,Trainer, TrainingArguments
import numpy as np
import evaluate
from consts import LABEL2ID_M, ID2LABEL_M, MODEL_CONFIGS
from tokenizer import get_tokenizer
import os

class HateClassifier():
    def __init__(self, model_path:str, language:str, is_multi_lang_model=False):
        self.tokenizer  = get_tokenizer(language, is_multi_lang_model)
        self.output_dir = os.path.join("hate", language, model_path)
        self.model      = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            config=MODEL_CONFIGS[model_path],
            num_labels=7,
            id2label=ID2LABEL_M,
            label2id=LABEL2ID_M,
        )

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