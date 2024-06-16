from transformers import AutoTokenizer
from functools import partial
from consts import TOKENIZER_CONFIG, MODELS


def get_tokenizer(language:str):
    """
    Returns the tokenizer used in the NLP models.

    Returns:
        AutoTokenizer: The tokenizer used in the NLP models.
    """
    tokenizer_path = MODELS[language]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        config=TOKENIZER_CONFIG,
    )
    return tokenizer


def generic_tokenizing_function(tokenizer: AutoTokenizer, examples):
    """
    Tokenizes the test_case column in the examples dataset.

    Args:
        examples (Dataset): The dataset containing the test_case column.

    Returns:
        Dataset: The dataset with the tokenized test_case column.
    """

    return tokenizer(examples["test_case"], truncation=True,  padding='max_length', max_length=150)


def get_tokenizing_function(tokenizer: AutoTokenizer):
    """
    Returns the tokenizing function used in the NLP models.

    Returns:
        function: The tokenizing function used in the NLP models.
    """
    tokenizer = get_tokenizer(tokenizer)
    return partial(generic_tokenizing_function, tokenizer)