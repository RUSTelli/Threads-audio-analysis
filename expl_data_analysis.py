import pandas as pd

def max_text_len(csv_file):
    df = pd.read_csv(csv_file)
    max_length = df['test_case'].str.len().max()
    print("Maximum length of text:", max_length)


def avg_text_len(csv_file):
    df = pd.read_csv(csv_file)

    # Calculate the average text length in the "test_case" column
    avg_len = df['test_case'].str.len().mean()
    print(f"Average text length: {avg_len}")


def count_label_occurrences(csv_file):
    # Read the dataset into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Extract the second column
    column_data = df['functionality']
    
    # Calculate frequency of each category
    category_counts = column_data.value_counts()
    
    # Print statistics
    print("functionalities occurences:")
    for category, count in category_counts.items():
        print(f"{category}: {count}")


def avg_text_len_by_label(csv_file):
    # group rows by "functionality" column and calculate the average text length in the "test_case" column for each group
    df = pd.read_csv(csv_file)
    avg_len_by_label = df.groupby("functionality")["test_case"].apply(lambda x: x.str.len().mean())
    print("Average text length by label:")
    print(avg_len_by_label)


def eda_pipeline(csv_file):
    max_text_len(csv_file)
    avg_text_len(csv_file)
    avg_text_len_by_label(csv_file)
    count_label_occurrences(csv_file)

eda_pipeline("test.csv")