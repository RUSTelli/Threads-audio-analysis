
from transformers import AutoTokenizer

class Tokenizer():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "osiria/distilbert-base-italian-cased",
            config="tokenizer_config.json",
        )

    def tokenize(self, text):
        return self.tokenizer(text, padding="max_length", truncation=True)["input_ids"]