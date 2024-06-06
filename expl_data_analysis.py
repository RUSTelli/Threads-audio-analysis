import pandas as pd
import csv

def print_categorical_stats(input_file):
    # Read the dataset into a DataFrame
    df = pd.read_csv(input_file)
    
    # Extract the second column
    column_data = df.iloc[:, 1]
    
    # Calculate frequency of each category
    category_counts = column_data.value_counts()
    
    # Print statistics
    print("Statistics for the second column:")
    for category, count in category_counts.items():
        print(f"{category}: {count}")

def max_text_length_in_first_column(csv_file):
    max_length = 0
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) > 0:  # Ensure row has at least one element
                max_length = max(max_length, len(row[0]))
    print("Maximum length of text in the first column:", max_length)



def main():
    input_file = 'output_dataset.csv'  # Change this to your dataset file path
    print_categorical_stats(input_file)
    max_text_length_in_first_column(input_file)

    


if __name__ == "__main__":
    main()
