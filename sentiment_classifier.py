from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn, argmax
import os
import numpy as np
import evaluate
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
        self.lang            = "multi" if is_multi_lang_model else language
        self.output_dir = os.path.join("sentiment", self.lang)

    def train_and_test(self, train_dataset, eval_dataset, test_dataset, epochs=2, batch_size=16):
        def _compute_metrics(eval_pred):
            metrics        = evaluate.combine(["accuracy", "f1", "precision", "recall"])
            logits, labels = eval_pred
            predictions    = np.argmax(logits, axis=-1)
            return metrics.compute(predictions=predictions, references=labels)

        def _compute_confusion_matrix(predictions, save_as_png=True):
            # Calculate confusion matrix
            labels = predictions.label_ids
            preds = predictions.predictions.argmax(-1)
            cm = confusion_matrix(labels, preds)

            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", annot_kws={"fontsize": 20})
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.title("Confusion Matrix for "+ self.lang+" Hate Speech Classifier: "+str(epochs)+" epochs, batch size "+ str(batch_size), pad=20, fontsize=16)
            plt.show()

            # Save confusion matrix as PNG
            if(save_as_png):
                cm_filename = os.path.join(self.output_dir, "confusion_matrix_"+self.lang+".png")
                plt.savefig(cm_filename)
                plt.close()
            else: 
                plt.show()

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
        predictions = trainer.predict(test_dataset)

        _compute_confusion_matrix(predictions)
        
        return predictions
    
    def predict(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")
        outputs = self.model(**inputs)
        probs = nn.functional.softmax(outputs.logits, dim=-1)
        y = argmax(probs, dim=1)
        return ID2LABEL_B[y.item()]

