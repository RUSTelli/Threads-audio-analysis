from transformers import AutoModelForSequenceClassification,Trainer, TrainingArguments
import numpy as np
import evaluate
from consts import LABEL2ID_B, ID2LABEL_B, LABEL2ID_M, ID2LABEL_M, MODEL_CONFIGS
from tokenizer import get_tokenizer

class SentimentClassifier():
    def __init__(self, model_path:str, language:str):
        self.tokenizer  = get_tokenizer(model_path)
        self.num_labels = 2
        self.output_dir = f"sentiment_{language}"
        self.model      = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            # TODO: implement a way of getting the config for multi-language model
            config=MODEL_CONFIGS[language],
            num_labels=self.num_labels,
            id2label=ID2LABEL_B,
            label2id=LABEL2ID_B,
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

    def predict(self, test_dataset):
        pass

# class HateClassifier():
#     def __init__(self, model_name:str):
#         self.tokenizer  = TOKENIZER
#         self.num_labels = 7
#         self.output_dir = f"hate_{model_name}"
#         self.model      = AutoModelForSequenceClassification.from_pretrained(
#             model_name, 
#             config="nlp_config.json",
#             num_labels=self.num_labels,
#             id2label=ID2LABEL_M,
#             label2id=LABEL2ID_M,
#         )


#     def train(self, dataset, epochs=2, batch_size=16):
#         def _compute_metrics(eval_pred):
#             # TODO: change metrics
#             metrics        = evaluate.combine(["accuracy", "f1", "precision", "recall"])
#             logits, labels = eval_pred
#             predictions    = np.argmax(logits, axis=-1)
#             return metrics.compute(predictions=predictions, references=labels)
    
#         training_args = TrainingArguments(
#             output_dir=self.output_dir,
#             learning_rate=2e-5,
#             per_device_train_batch_size=batch_size,
#             per_device_eval_batch_size=batch_size,
#             num_train_epochs=epochs,
#             weight_decay=0.01,
#             eval_strategy="epoch",
#             save_strategy="epoch",
#             load_best_model_at_end=True,
#             push_to_hub=False,
#         )

#         trainer = Trainer(
#             model=self.model,
#             args=training_args,
#             train_dataset=dataset["train"],
#             eval_dataset=dataset["test"],
#             tokenizer=self.tokenizer,
#             compute_metrics=_compute_metrics,
#         )

#         trainer.train()

#     def predict(self, test_dataset):
#         pass