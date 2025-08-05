import pandas as pd
import yaml
import numpy as np

# load the config
def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        return None

def load_data(config):
    try:
        data = pd.read_csv(config['data']['file_paths']['merged_data'])
        return data
    except FileNotFoundError:
        print(f"Data file not found: {config['data']['file_paths']['merged_data']}")
        return None
    
# create target variable
def map_loan_risk(loan_status):
    # Bad loan statuses (risky = 1)
    bad_loan_statuses = [
        'External Collection', 
        'Internal Collection', 
        'Returned Item', 
        'Charged Off', 
        'Charged Off Paid Off', 
        'Settled Bankruptcy', 
        'Settlement Paid Off', 
        'Settlement Pending Paid Off'
    ]
    
    # Good loan statuses (risky = 0)
    good_loan_statuses = [
        'Paid Off Loan', 
        'New Loan',
        'Pending Paid Off'
    ]
    
    if loan_status in bad_loan_statuses:
        return 1  # Risky loan
    elif loan_status in good_loan_statuses:
        return 0  # Good loan
    else:
        return np.nan  # Excluded cases
    
def create_target_variable(data):
    # Apply the mapping
    data['isRisky'] = data['loanStatus'].apply(map_loan_risk)

    # drop the loanStatus column as it is no longer needed
    data = data.drop(columns=['loanStatus'])

    # remove rows with NaN in 'isRisky' as they do not contribute to the target variable
    data = data.dropna(subset=['isRisky'])
    return data

# remove identifiers
def remove_identifiers(data):
    identifiers = ['loanId', 'clarityFraudId', 'underwritingid', 'anon_ssn']
    data = data.drop(columns=identifiers)
    return data

# data feature engineering
def feature_engineering(data):
    # find the days difference between applicationDate and originatedData
    data['days_diff'] = (pd.to_datetime(data['originatedDate'], format='mixed') - pd.to_datetime(data['applicationDate'], format='mixed')).dt.days

    # Or fill with -1 to indicate missing originated date
    data['days_diff'] = data['days_diff'].fillna(-1)

    # Convert applicationDate to datetime if not already done
    data['applicationDate'] = pd.to_datetime(data['applicationDate'], format='mixed')

    # Extract time-based components
    data['application_month'] = data['applicationDate'].dt.month
    data['application_day_of_week'] = data['applicationDate'].dt.dayofweek  # Monday=0, Sunday=6
    data['application_day_of_month'] = data['applicationDate'].dt.day
    data['application_week_of_year'] = data['applicationDate'].dt.isocalendar().week

    # Create boolean flags
    data['is_month_start'] = data['application_day_of_month'] <= 3  # First 3 days of month
    data['is_month_end'] = data['application_day_of_month'] >= 28   # Last few days (28+ to handle Feb)
    data['is_weekend'] = data['application_day_of_week'].isin([5, 6])  # Saturday=5, Sunday=6

    # Create binary feature for originated loans
    data['is_originated'] = data['originatedDate'].notna()

    data = data.drop(columns="originatedDate")  # Drop the original originatedDate column
    return data

# imputation 
def impute_missing_values(data):
    df_processed = data.copy()

    # Identify numerical and categorical columns
    numerical_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove target variable and date from processing if it exists
    if 'isRisky' in numerical_columns:
        numerical_columns.remove('isRisky')
    
    # Handle numerical columns
    print("\n--- NUMERICAL COLUMNS ---")
    for col in numerical_columns:
        missing_count = df_processed[col].isnull().sum()
        if missing_count > 0:
            missing_pct = (missing_count / len(df_processed)) * 100
            print(f"{col}: {missing_count} missing ({missing_pct:.2f}%)")
            
            # Create missing indicator
            indicator_col = f"{col}_is_missing"
            df_processed[indicator_col] = df_processed[col].isnull().astype(int)
            
            # Impute with median
            median_value = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_value)
            
            print(f"  → Created indicator: {indicator_col}")
            print(f"  → Imputed with median: {median_value:.4f}")
        else:
            print(f"{col}: No missing values")
    
    # Handle categorical columns
    print("\n--- CATEGORICAL COLUMNS ---")
    for col in categorical_columns:
        missing_count = df_processed[col].isnull().sum()
        if missing_count > 0:
            missing_pct = (missing_count / len(df_processed)) * 100
            print(f"{col}: {missing_count} missing ({missing_pct:.2f}%)")
            
            # Impute with 'missing' category
            df_processed[col] = df_processed[col].fillna('missing')
            print(f"  → Imputed with 'missing' category")
        else:
            print(f"{col}: No missing values")
    
    # Summary
    remaining_missing = df_processed.isnull().sum().sum()
    print(f"\n=== SUMMARY ===")
    print(f"Remaining missing values: {remaining_missing}")
    print(f"Original shape: {data.shape}")
    print(f"Processed shape: {df_processed.shape}")
    
    # Show new columns created
    new_columns = [col for col in df_processed.columns if col.endswith('_is_missing')]
    if new_columns:
        print(f"Missing indicator columns created: {len(new_columns)}")
        for col in new_columns:
            print(f"  - {col}")
    
    return df_processed

# save the processed data
def save_processed_data(data, config):
    output_path = config['data']['file_paths']['preprocessed_data']
    try:
        data.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
    except Exception as e:
        print(f"Error saving processed data: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Merging the data.')
    parser.add_argument('--config', required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    config = load_config(args.config)
    data = load_data(config)
    # create target variable
    data = create_target_variable(data)
    # remove identifiers
    data = remove_identifiers(data)
    # feature engineering
    data = feature_engineering(data)
    # impute missing values
    data = impute_missing_values(data)
    # save processed data
    save_processed_data(data, config)

# how to run
# py -3.12 src/preprocessing/preprocess.py --config "config/config.yaml"