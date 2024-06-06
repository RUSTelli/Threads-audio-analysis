from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from consts import CUSTOM_FUNCTIONALITIES
import pandas as pd

def drop_typo_rows(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    # Filter out rows where 'functionality' column contains the word "spell"
    df_filtered = df[~df['functionality'].str.contains("spell", case=False, na=False)]
    # Save the filtered DataFrame back to the same file
    df_filtered.to_csv(csv_file, index=False)


def drop_useless_columns(csv_file):
    # drop all the column which are not "functionality", "text_case" and "target_ident"
    df = pd.read_csv(csv_file)
    df = df[["functionality", "test_case", "target_ident"]]
    df.to_csv(csv_file, index=False) 


def update_functionalities(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Iterate over the rows of the DataFrame
    for index, row in df.iterrows():
        # Check each functionality set in CUSTOM_FUNCTIONALITIES
        for new_func, old_funcs in CUSTOM_FUNCTIONALITIES.items():
            # If the row's 'functionality' is in the set, replace it
            if row['functionality'] in old_funcs:
                df.at[index, 'functionality'] = new_func
                break  # Stop checking other sets once a match is found

    # Save the updated DataFrame back to the same file
    df.to_csv(csv_file, index=False)


def encode_labels(csv_file):
    df = pd.read_csv(csv_file)
    labels = df['functionality']    
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()
    # Fit and transform labels
    encoded_labels = label_encoder.fit_transform(labels)
    # Replace original labels with encoded labels in the DataFrame
    df['functionality'] = encoded_labels
    # Save the DataFrame with encoded labels to a new CSV file
    df.to_csv(csv_file, index=False)


def split_train_val_test(csv_file, test_size=0.2, validation_size=0.1, random_state=42):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    y = df['functionality']
    X = df['test_case']

    # First split the data into training and temp sets
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    # Then split the temp set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=validation_size/(1-test_size), stratify=y_train_temp, random_state=random_state)

    # Concatenate features and labels for each set
    training_dataset = pd.concat([X_train, y_train], axis=1)
    validation_dataset = pd.concat([X_val, y_val], axis=1)
    test_dataset = pd.concat([X_test, y_test], axis=1)

    # Write to CSV files
    training_dataset.to_csv("training_dataset.csv", index=False)
    validation_dataset.to_csv("validation_dataset.csv", index=False)
    test_dataset.to_csv("test_dataset.csv", index=False)




def data_manipulation_pipeline(dataset):
    drop_typo_rows(dataset)
    drop_useless_columns(dataset)
    update_functionalities(dataset)
    encode_labels(dataset)
    split_train_val_test(dataset)


data_manipulation_pipeline("test.csv")