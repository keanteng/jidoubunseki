import pandas as pd
import yaml
import great_expectations as gx
import warnings

# turn off warnings
warnings.filterwarnings("ignore")

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
        data = pd.read_csv(config['data']['file_paths']['loan_data'])
        return data
    except FileNotFoundError:
        print(f"Loan data file not found: {config['data']['file_paths']['loan_data']}")
        return None

def load_payment_data(config):
    try:
        data = pd.read_csv(config['data']['file_paths']['payment_data'])
        return data
    except FileNotFoundError:
        print(f"Payment data file not found: {config['data']['file_paths']['payment_data']}")
        return None

def load_clarity_underwriting_variables(config):
    try:
        data = pd.read_csv(config['data']['file_paths']['clarity_underwriting_variables'], low_memory=False)
        return data
    except FileNotFoundError:
        print(f"Clarity underwriting variables file not found: {config['data']['file_paths']['clarity_underwriting_variables']}")
        return None
    
def merge_data(loan_data, payment_data, clarity_data):
    # merge the payment with loan first
    loan_payment = pd.merge(loan_data, payment_data, on='loanId', how='left')

    # merge the clarity data with the loan_payment
    loan_payment = pd.merge(loan_payment, clarity_data, left_on='clarityFraudId', right_on='underwritingid', how='left')

    return loan_payment

def save_merged_data(merged_data, config):
    output_path = config['data']['file_paths']['merged_data']
    try:
        merged_data.to_csv(output_path, index=False)
        print(f"Merged data saved to {output_path}")
    except Exception as e:
        print(f"Error saving merged data: {e}")

def validate_loan_data(data):
    """Validate loan data using Great Expectations"""
    try:
        context = gx.get_context(mode="ephemeral")
        
        data_asset = context.data_sources.add_pandas(name="loan_data_source").read_dataframe(
            data,
            asset_name="loan_dataframe_asset"
        )
        
        validator = context.get_validator(batch_request=data_asset.batch_request)
        
        # Define expectations
        validator.expect_column_to_exist('loanId')
        validator.expect_column_values_to_be_unique('loanId')
        validator.expect_column_to_exist('clarityFraudId')

        # Validate
        validation_result = validator.validate()
        
        if validation_result.success:
            print("✓ Loan data validation passed")
            return True
        else:
            print("✗ Loan data validation failed:")
            for result in validation_result.results:
                if not result.success:
                    # Corrected line: Access 'type' from expectation_config
                    print(f"  - {result.expectation_config.type}") 
            return False
            
    except Exception as e:
        print(f"Error validating loan data: {e}")
        return False

def validate_payment_data(data):
    """Validate payment data using Great Expectations"""
    try:
        context = gx.get_context(mode="ephemeral")
        
        data_asset = context.data_sources.add_pandas(name="payment_data_source").read_dataframe(
            data,
            asset_name="payment_dataframe_asset"
        )

        validator = context.get_validator(batch_request=data_asset.batch_request)

        # Define expectations
        validator.expect_column_to_exist('loanId')

        # Validate
        validation_result = validator.validate()

        if validation_result.success:
            print("✓ Payment data validation passed")
            return True
        else:
            print("✗ Payment data validation failed:")
            for result in validation_result.results:
                if not result.success:
                    # Corrected line: Access 'type' from expectation_config
                    print(f"  - {result.expectation_config.type}")
            return False

    except Exception as e:
        print(f"Error validating payment data: {e}")
        return False

def validate_clarity_data(data):
    """Validate clarity underwriting data using Great Expectations"""
    try:
        context = gx.get_context(mode="ephemeral")
        
        data_asset = context.data_sources.add_pandas(name="clarity_data_source").read_dataframe(
            data,
            asset_name="clarity_dataframe_asset"
        )

        validator = context.get_validator(batch_request=data_asset.batch_request)

        # Define expectations
        validator.expect_column_to_exist('underwritingid')
        validator.expect_column_values_to_not_be_null('underwritingid')
        validator.expect_column_values_to_be_unique('underwritingid')

        # Validate
        validation_result = validator.validate()

        if validation_result.success:
            print("✓ Clarity underwriting data validation passed")
            return True
        else:
            print("✗ Clarity underwriting data validation failed:")
            for result in validation_result.results:
                if not result.success:
                    # Corrected line: Access 'type' from expectation_config
                    print(f"  - {result.expectation_config.type}")
            return False
            
    except Exception as e:
        print(f"Error validating clarity data: {e}")
        return False

def validate_merged_data(data):
    """Validate merged data using Great Expectations"""
    try:
        context = gx.get_context(mode="ephemeral")
        
        data_asset = context.data_sources.add_pandas(name="merged_data_source").read_dataframe(
            data,
            asset_name="merged_dataframe_asset"
        )

        validator = context.get_validator(batch_request=data_asset.batch_request)

        # Define expectations
        validator.expect_column_to_exist('loanId')
        validator.expect_column_to_exist('clarityFraudId')
        validator.expect_column_to_exist('underwritingid')

        # Validate
        validation_result = validator.validate()

        if validation_result.success:
            print("✓ Merged data validation passed")
            return True
        else:
            print("✗ Merged data validation failed:")
            for result in validation_result.results:
                if not result.success:
                    # Corrected line: Access 'type' from expectation_config
                    print(f"  - {result.expectation_config.type}")
            return False
    except Exception as e:
        print(f"Error validating merged data: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Merging the data.')
    parser.add_argument('--config', required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    config = load_config(args.config)
    loan_data = load_loan_data(config)
    payment_data = load_payment_data(config)
    clarity_data = load_clarity_underwriting_variables(config)

    # Validate individual datasets before merging
    print("Validating datasets...")
    
    if not all([
        validate_loan_data(loan_data),
        validate_payment_data(payment_data),
        validate_clarity_data(clarity_data)
    ]):
        print("❌ Data validation failed. Stopping merge process.")
        exit(1)

    print("✅ All datasets passed validation. Proceeding with merge...")
    
    # Merge the data
    merged_data = merge_data(loan_data, payment_data, clarity_data)

    # Validate merged data
    if validate_merged_data(merged_data):
        print("✅ Merged data validation passed.")
        # Save the merged data
        save_merged_data(merged_data, config)
    else:
        print("❌ Merged data validation failed. Not saving results.")
        exit(1)  # Exit with error code to fail the GitHub workflow

# how to run
# py -3.12 src/data/merge.py --config "config/config.yaml"