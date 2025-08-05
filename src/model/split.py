import pandas as pd
import yaml

# load the config
def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        return None

def load_loan_data(config):
    try:
        data = pd.read_csv(config['data']['file_paths']['preprocessed_data'], low_memory=False)
        return data
    except FileNotFoundError:
        print(f"Loan data file not found: {config['data']['file_paths']['preprocessed_data']}")
        return None
    
def time_split_data(data):
    # Sort by date to understand chronological distribution
    loan_sorted = data.sort_values('applicationDate')

    # Calculate splitting points (common ratios: 70% train, 15% val, 15% test)
    total_records = len(loan_sorted)
    train_size = int(0.7 * total_records)
    val_size = int(0.15 * total_records)

    # Find the actual dates at these split points
    train_end_date = loan_sorted.iloc[train_size]['applicationDate']
    val_end_date = loan_sorted.iloc[train_size + val_size]['applicationDate']

    print(f"\nSuggested split dates:")
    print(f"Train period: {loan_sorted['applicationDate'].min()} to {train_end_date}")
    print(f"Validation period: {train_end_date} to {val_end_date}")
    print(f"Test period: {val_end_date} to {loan_sorted['applicationDate'].max()}")

    # Show data distribution
    print(f"\nData distribution:")
    print(f"Train set: {train_size} records ({train_size/total_records*100:.1f}%)")
    print(f"Validation set: {val_size} records ({val_size/total_records*100:.1f}%)")
    print(f"Test set: {total_records - train_size - val_size} records ({(total_records - train_size - val_size)/total_records*100:.1f}%)")

    # Create the actual splits
    train_data = loan_sorted[loan_sorted['applicationDate'] <= train_end_date]
    val_data = loan_sorted[(loan_sorted['applicationDate'] > train_end_date) & 
                        (loan_sorted['applicationDate'] <= val_end_date)]
    test_data = loan_sorted[loan_sorted['applicationDate'] > val_end_date]

    # remove the applicationDate column as it is no longer needed
    train_data = train_data.drop(columns=['applicationDate'])
    val_data = val_data.drop(columns=['applicationDate'])
    test_data = test_data.drop(columns=['applicationDate'])

    print(f"\nActual split sizes:")
    print(f"Train: {len(train_data)} records")
    print(f"Validation: {len(val_data)} records") 
    print(f"Test: {len(test_data)} records")

    return train_data, val_data, test_data

# save the split data to CSV files
def save_train(data, config):
    path = config['data']['file_paths']['train_data']
    try:
        data.to_csv(path, index=False)
        print(f"Train data saved to {path}")
    except Exception as e:
        print(f"Error saving train data: {e}")

def save_val(data, config):
    path = config['data']['file_paths']['valid_data']
    try:
        data.to_csv(path, index=False)
        print(f"Validation data saved to {path}")
    except Exception as e:
        print(f"Error saving validation data: {e}")

def save_test(data, config):
    path = config['data']['file_paths']['test_data']
    try:
        data.to_csv(path, index=False)
        print(f"Test data saved to {path}")
    except Exception as e:
        print(f"Error saving test data: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Merging the data.')
    parser.add_argument('--config', required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    config = load_config(args.config)
    data = load_loan_data(config)
    train_data, val_data, test_data = time_split_data(data)

    # Save the split data
    save_train(train_data, config)
    save_val(val_data, config)
    save_test(test_data, config)

# how to run
# py -3.12 src/model/split.py --config "config/config.yaml"