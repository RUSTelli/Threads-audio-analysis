from transformers import AutoTokenizer
from functools import partial
from consts import MODELS, MAX_SEQ_LEN


def get_tokenizer(language:str, is_multi_lang_model = False):
    """
    Returns the tokenizer used in the NLP models.

    Returns:
        AutoTokenizer: The tokenizer used in the NLP models.
    """
    tokenizer_path = MODELS["multi"] if  is_multi_lang_model else MODELS[language]
    return AutoTokenizer.from_pretrained(tokenizer_path)


def _generic_tokenizing_function(tokenizer: AutoTokenizer, examples):
    """
    Serves as base for tokenizing function (see below).

    Args:
        examples (Dataset): The dataset row.

    Returns:
        Dataset: the row with the tokenized test_case column.
    """

    return tokenizer(examples["test_case"], truncation=True,  padding='max_length', max_length=MAX_SEQ_LEN)


def get_tokenizing_function(language: str, multi_language_model = False):
    """
    Returns the specif-language tokenizing function used in data preparation.

    Returns:
        function: Callable tokenizing function.
    """
    tokenizer = get_tokenizer(language, multi_language_model)
    return partial(_generic_tokenizing_function, tokenizer)