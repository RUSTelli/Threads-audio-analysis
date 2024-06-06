from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, set_seed
from transformers import AutoModel
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch

def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)

    mse = mean_squared_error(labels, logits)
    rmse = mean_squared_error(labels, logits, squared=False)
    mae = mean_absolute_error(labels, logits)
        
    return {"mse": mse, "rmse": rmse, "mae": mae}


NUM_OF_EPOCHS = 2
MAX_INPUT_LEN = 100

MODEL_CKPT = "osiria/distilbert-base-italian-cased"                             
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_LABELS = 1                                                                  # TODO probably must be set to 4
BATCH_SIZE = 64                                                                 # TODO probably must be set to 100 (short dataset)

MODEL_NAME = f"{MODEL_CKPT}-ThreadsAudioHateAnalysis"
LEARNING_RATE = 1.5e-5

set_seed(42)


model = (AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT, num_labels=NUM_LABELS)).to(DEVICE)

training_args = TrainingArguments(
    output_dir=MODEL_NAME,
    num_train_epochs=NUM_OF_EPOCHS, 
    per_device_train_batch_size=BATCH_SIZE, 
    per_device_eval_batch_size=BATCH_SIZE, 
    weight_decay=0.01,
    learning_rate=LEARNING_RATE,
    use_cpu=True,                                                        
    eval_strategy='epoch',                                         
    save_total_limit=10,
    logging_strategy='epoch',
    load_best_model_at_end=True,
    disable_tqdm=False,
    metric_for_best_model='mae',
    greater_is_better=False,
    save_strategy='epoch',
    push_to_hub=True
)

train_dataset = ''
test_dataset = ''
eval_dataset = ''



trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=encoded_ds['train'], 
    eval_dataset=encoded_ds['eval'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_for_regression
)

trainer.train()

#trainer.eval_dataset = encoded_ds['test']
trainer.evaluate()
