from datasets import load_dataset
from consts import HATE_TYPES, BINARY_CLUSTER, DATASETS, MODELS
from tokenizer import get_tokenizing_function

def _aggregate_functionalities(row):
    """
    Aggregates the functionalities into sentiments.

    Args:
        row (dict): The row of the dataset.

    Returns:
        dict: The modified row with aggregated functionality.
    """
    functionality = row['functionality']
    for sentiment, functionalities in BINARY_CLUSTER.items():
        if functionality in functionalities:
            row['functionality'] = sentiment
    return row

def _aggregate_hate_types(row):
    """
    Aggregates the hate types.

    Args:
        row (dict): The row of the dataset.

    Returns:
        dict: The modified row with aggregated hate types.
    """
    target_identity = row['target_ident']
    row['target_ident'] = 'GENERIC'  
    for hate_type, targets in HATE_TYPES.items():
        if target_identity in targets:
            row['target_ident'] = hate_type
    return row

def _rename_columns(dataset, classification_type="sentiment"):
    """
    Renames the columns of the dataset.

    Args:
        dataset (Dataset): The dataset to rename columns.
        classification_type (str): The type of classification.

    Returns:
        Dataset: The dataset with renamed columns.
    """
    dataset = dataset.rename_column("test_case", "text")
    if classification_type == "sentiment":
        dataset = dataset.rename_column("functionality", "label")
    elif classification_type == "multi":
        dataset = dataset.rename_column("target_ident", "label")
    return dataset

def data_pipeline(classification_type="sentiment", language="it"):
    # load dataset
    dataset = load_dataset(DATASETS[language], split="test")
    # drop rows with typos in the text
    dataset = dataset.filter(lambda example: "spell" not in example["functionality"])
    # keep only the necessary columns
    dataset = dataset.select_columns(["functionality", "test_case", "target_ident"])
    # aggregate functionalities into sentiments (hateful, non_hateful)
    dataset = dataset.map(_aggregate_functionalities)
    # tokenize text
    tokenizing_function = get_tokenizing_function(language)
    dataset = dataset.map(tokenizing_function)
    # multi-class classification operations
    if classification_type == "multi":
        #drop NON_HATEFUL rows 
        dataset = dataset.filter(lambda example: example["functionality"] == "HATEFUL")
        # aggregate hate types
        dataset = dataset.map(_aggregate_hate_types)
    # rename columns    
    dataset = _rename_columns(dataset, classification_type=classification_type)
    # encode labels
    dataset = dataset.class_encode_column("label")
    # split the dataset into training, validation and test with a fixed seed of 42
    dataset = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")

    return dataset