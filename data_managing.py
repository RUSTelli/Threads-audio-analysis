import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def cluster_LGBT_labels(csv_file, output_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Define a function to replace "gay" or "trans" with "LGBT"
    def replace_label(label):
        if 'gay' in label.lower() or 'trans' in label.lower():
            return 'LGBT'
        return label

    # Apply the function to the second column
    df.iloc[:, 1] = df.iloc[:, 1].apply(replace_label)

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_file, index=False)


def encode_labels(csv_file):
    # Load the CSV file into a DataFrame, skipping the first row as it contains headers
    df = pd.read_csv(csv_file, skiprows=1)

    # Extract labels
    labels = df.iloc[:, 1]  # Assuming the second column is the label

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Fit and transform labels
    encoded_labels = label_encoder.fit_transform(labels)

    # Replace original labels with encoded labels in the DataFrame
    df.iloc[:, 1] = encoded_labels

    # Save the DataFrame with encoded labels to a new CSV file
    df.to_csv("encoded_" + csv_file, index=False)


def split_csv(csv_file, output_folder, test_size=0.2, validation_size=0.1, random_state=42):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Stratified split based on the second column
    X = df.iloc[:, 0]  # Assuming the first column is the feature
    y = df.iloc[:, 1]  # Assuming the second column is the label

    # First split the data into training and temp sets
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    # Then split the temp set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=validation_size/(1-test_size), stratify=y_train_temp, random_state=random_state)

    # Concatenate features and labels for each set
    training_dataset = pd.concat([X_train, y_train], axis=1)
    validation_dataset = pd.concat([X_val, y_val], axis=1)
    test_dataset = pd.concat([X_test, y_test], axis=1)

    # Write to CSV files
    training_dataset.to_csv(output_folder + "/training_dataset.csv", index=False)
    validation_dataset.to_csv(output_folder + "/validation_dataset.csv", index=False)
    test_dataset.to_csv(output_folder + "/test_dataset.csv", index=False)




#encode_labels("synthetic.csv")
#split_csv("encoded_synthetic.csv", ".")
cluster_LGBT_labels("hate_lgbt.csv", "output_dataset.csv")