from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained(
    "osiria/distilbert-base-italian-cased",
    config="configs/tokenizer_config.json",
)

def tokenizing_function(examples):
    """
    Tokenizes the test_case column in the examples dataset.

    Args:
        examples (Dataset): The dataset containing the test_case column.

    Returns:
        Dataset: The dataset with the tokenized test_case column.
    """
    return TOKENIZER(examples["test_case"], truncation=True,  padding='max_length', max_length=150)