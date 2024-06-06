import pandas as pd

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

def main():
    input_file = 'synthetic.csv'  # Change this to your dataset file path
    print_categorical_stats(input_file)

if __name__ == "__main__":
    main()
