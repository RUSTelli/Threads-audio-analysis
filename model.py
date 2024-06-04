from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, set_seed
from transformers import AutoModel
import torch


NUM_OF_EPOCHS = 2
MAX_INPUT_LEN = int(round(150*1.3, 0)) # TODO EDIT

MODEL_CKPT = "osiria/distilbert-base-italian-cased" # TODO change to distilbert ita
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_LABELS = 1 # TODO probably must be set to 4
BATCH_SIZE = 64 # TODO probably must be set to 100 (short dataset)

MODEL_NAME = f"{MODEL_CKPT}-ThreadsAudioHateAnalysis"
LEARNING_RATE = 1.5e-5

set_seed(42)


model = (AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT, num_labels=NUM_LABELS)).to(DEVICE)