from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn, argmax
import os
from consts import LABEL2ID_M, ID2LABEL_M, MAX_SEQ_LEN
from tokenizer import get_tokenizer


class HateClassifier():
    def __init__(self, model_path:str, language:str, is_multi_lang_model=False):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=8,
            id2label=ID2LABEL_M,
            label2id=LABEL2ID_M,
        )
        self.tokenizer  = get_tokenizer(language, is_multi_lang_model)
        self.lang       = "multi" if is_multi_lang_model else language
        self.output_dir = os.path.join("hate", self.lang)
        

    def train_and_test(self, train_dataset, eval_dataset, test_dataset, epochs=1, batch_size=16):
        def _compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            
            # Calculate accuracy
            accuracy = accuracy_score(labels, preds)

            # Calculate precision, recall, and F1-score
            precision = precision_score(labels, preds, average='micro')
            recall = recall_score(labels, preds, average='micro', zero_division="warn")
            f1 = f1_score(labels, preds, average='micro')
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

        def _compute_confusion_matrix(predictions, save_as_png=True):

            # Calculate confusion matrix
            labels = predictions.label_ids
            preds = predictions.predictions.argmax(-1)
            cm = confusion_matrix(labels, preds)
            
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=list(ID2LABEL_M.values()), yticklabels=list(ID2LABEL_M.values()))
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.xticks(rotation=45)
            plt.title("Confusion Matrix for "+ self.lang+" Hate Speech Classifier: "+str(epochs)+" epochs, batch size "+ str(batch_size), pad=20, fontsize=16) 
            
            # Save or show confusion matrix
            if(save_as_png):
                cm_filename = os.path.join(self.output_dir, "confusion_matrix_"+self.lang+".png")
                plt.savefig(cm_filename)
                plt.close()
            else: 
                plt.show()

            return cm

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
        return ID2LABEL_M[y.item()]