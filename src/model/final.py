import pandas as pd
import yaml
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, classification_report, average_precision_score
import joblib
import mlflow
import warnings
import os

# disable warnings for cleaner output
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

def get_project_root():
    """Get the project root directory"""
    # Assuming this file is in src/model/ and project root is 2 levels up
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current_dir))

def resolve_path(config_path, project_root=None):
    """Resolve path to be relative to project root"""
    if project_root is None:
        project_root = get_project_root()
    
    # If it's already an absolute path, return as is for backward compatibility
    if os.path.isabs(config_path):
        return config_path
    
    # Otherwise, make it relative to project root
    return os.path.join(project_root, config_path)

def load_loan_data(config):
    project_root = get_project_root()
    try:
        train_path = resolve_path(config['data']['file_paths']['train_data'], project_root)
        valid_path = resolve_path(config['data']['file_paths']['valid_data'], project_root)
        test_path = resolve_path(config['data']['file_paths']['test_data'], project_root)
        
        train_data = pd.read_csv(train_path, low_memory=False)
        valid_data = pd.read_csv(valid_path, low_memory=False)
        test_data = pd.read_csv(test_path, low_memory=False)
        return train_data, valid_data, test_data
    except FileNotFoundError as e:
        print(f"Loan data file not found: {e}")
        return None

def load_best_params(config):
    project_root = get_project_root()
    try:
        params_path = resolve_path(config['data']['model_paths']['best_params'], project_root)
        best_params = joblib.load(params_path)
        return best_params
    except FileNotFoundError:
        params_path = resolve_path(config['data']['model_paths']['best_params'], project_root)
        print(f"Best parameters file not found: {params_path}")
        return None

def setup_mlflow(config):
    """Setup MLflow tracking for Databricks with Unity Catalog"""
    import os
    
    # Check if we're using Databricks
    tracking_uri = config.get('mlflow', {}).get('tracking_uri', 'databricks')
    
    if tracking_uri == 'databricks':
        # For Databricks, check environment variables
        databricks_host = os.getenv('DATABRICKS_HOST')
        databricks_token = os.getenv('DATABRICKS_TOKEN')
        mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
        
        if not databricks_host or not databricks_token:
            print("❌ Error: DATABRICKS_HOST and DATABRICKS_TOKEN environment variables must be set")
            print("Please run:")
            print('export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"')
            print('export DATABRICKS_TOKEN="your-token"')
            print('export MLFLOW_TRACKING_URI="databricks"')
            raise ValueError("Missing Databricks credentials")
        
        print(f"✅ Using Databricks MLflow tracking")
        print(f"   Host: {databricks_host}")
        print(f"   Tracking URI: {mlflow_tracking_uri}")
        
        # Set tracking URI for Databricks
        mlflow.set_tracking_uri("databricks")
        
        # For Unity Catalog, set registry URI to databricks-uc
        mlflow.set_registry_uri("databricks-uc")
        print("✅ Using Unity Catalog Model Registry")
        
    else:
        # ...existing code...
        project_root = get_project_root()
        
        if tracking_uri.startswith('file:'):
            path_part = tracking_uri.replace('file:', '')
            if not os.path.isabs(path_part):
                path_part = path_part.lstrip('./')
                absolute_path = os.path.join(project_root, path_part)
                tracking_uri = f"file:{absolute_path}"
        elif not os.path.isabs(tracking_uri):
            absolute_path = os.path.join(project_root, tracking_uri)
            tracking_uri = f"file:{absolute_path}"
        
        mlflow.set_tracking_uri(tracking_uri)
    
    # Set experiment name
    experiment_name = config.get('mlflow', {}).get('experiment_name', '/Shared/loan_risk_prediction')
    
    try:
        mlflow.set_experiment(experiment_name)
    except mlflow.exceptions.MlflowException as e:
        if "deleted experiment" in str(e):
            print(f"Warning: Experiment '{experiment_name}' was deleted. Creating new experiment with timestamp.")
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            new_experiment_name = f"{experiment_name}_{timestamp}"
            mlflow.set_experiment(new_experiment_name)
            experiment_name = new_experiment_name
        else:
            raise e
    
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"MLflow registry URI: {mlflow.get_registry_uri()}")
    print(f"MLflow experiment: {experiment_name}")

# final training using best params
# Prepare the data for LightGBM
def prepare_lgb_data(train_data, val_data, test_data, target_col='isRisky'):
    """
    Prepare data for LightGBM training with proper categorical feature handling
    """
    # Separate features and target
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    
    X_val = val_data.drop(columns=[target_col])
    y_val = val_data[target_col]
    
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]
    
    # Identify categorical columns (object and category types)
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Categorical columns found: {len(categorical_columns)}")
    for col in categorical_columns:
        
        # Convert to category dtype for LightGBM's native support
        X_train[col] = X_train[col].astype('category')
        X_val[col] = X_val[col].astype('category')
        X_test[col] = X_test[col].astype('category')
    
    print(f"\nFinal data shapes:")
    print(f"Train: {X_train.shape}, Target: {y_train.shape}")
    print(f"Validation: {X_val.shape}, Target: {y_val.shape}")
    print(f"Test: {X_test.shape}, Target: {y_test.shape}")
    
    # Verify data types
    print(f"\nData types check:")
    object_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        print(f"WARNING: Object columns still present: {object_cols}")
    else:
        print("✅ No object columns remaining")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, categorical_columns

def register_and_promote_model(model_name, run_id, test_auc, val_auc, config):
    """
    Register the model with versioning (simplified for existing Databricks workspace)
    """
    client = mlflow.tracking.MlflowClient()

    # First verify the model artifact exists
    try:
        lgb_artifacts = client.list_artifacts(run_id, "lightgbm_model")
        if lgb_artifacts:
            model_uri = f"runs:/{run_id}/lightgbm_model"
            print(f"✅ Using lightgbm_model artifact path")
        else:
            print(f"❌ No lightgbm_model artifacts found for run {run_id}")
            return None
    except Exception as e:
        print(f"❌ Error checking artifacts: {e}")
        return None
    
    # Use existing workspace catalog instead of trying to create new one
    catalog = "workspace"  # Use existing catalog
    schema = "default"     # Use default schema
    model_base_name = model_name.split('.')[-1] if '.' in model_name else model_name
    full_model_name = f"{catalog}.{schema}.{model_base_name}"
    
    print(f"🎯 Registering model: {full_model_name}")
    print(f"   Model URI: {model_uri}")
    
    try:
        # Register the model in Unity Catalog
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=full_model_name
        )
        
        print(f"✅ Model registered in Unity Catalog as version {model_version.version}")
        
        # Set model version tags
        try:
            client.set_model_version_tag(
                name=full_model_name,
                version=model_version.version,
                key="description",
                value=f"LightGBM model trained with AUC: {test_auc:.4f}"
            )
            
            client.set_model_version_tag(
                name=full_model_name,
                version=model_version.version,
                key="test_auc",
                value=f"{test_auc:.4f}"
            )
            
            client.set_model_version_tag(
                name=full_model_name,
                version=model_version.version,
                key="val_auc",
                value=f"{val_auc:.4f}"
            )
            
            client.set_model_version_tag(
                name=full_model_name,
                version=model_version.version,
                key="training_date",
                value=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # Add model performance evaluation
            performance_status = evaluate_model_performance(test_auc, val_auc, config)
            
            # Add this section to use the promote_to_production function
            if performance_status == "PRODUCTION_READY":
                promote_to_production(client, full_model_name, model_version.version, test_auc)
            
            client.set_model_version_tag(
                name=full_model_name,
                version=model_version.version,
                key="performance_status",
                value=performance_status
            )
            
            print(f"✅ Model version {model_version.version} registered in Unity Catalog with performance status: {performance_status}")
            
        except Exception as e:
            print(f"❌ Error setting model tags: {e}")
            performance_status = evaluate_model_performance(test_auc, val_auc, config)
            print(f"✅ Model performance status: {performance_status}")
    
    except Exception as e:
        print(f"❌ Error registering model in Unity Catalog: {e}")
        return None
    
    return model_version, performance_status

def evaluate_model_performance(test_auc, val_auc, config):
    """
    Evaluate model performance against thresholds (without using deprecated staging)
    """
    min_test_auc = config.get('model_promotion', {}).get('min_test_auc', 0.82)
    min_val_auc = config.get('model_promotion', {}).get('min_val_auc', 0.72)
    
    print(f"\n=== MODEL PERFORMANCE EVALUATION ===")
    print(f"Test AUC: {test_auc:.4f} (Threshold: {min_test_auc})")
    print(f"Val AUC: {val_auc:.4f} (Threshold: {min_val_auc})")
    
    if test_auc >= min_test_auc and val_auc >= min_val_auc:
        status = "PRODUCTION_READY"
        print("✅ Model meets production criteria")
    elif test_auc >= min_test_auc * 0.9 and val_auc >= min_val_auc * 0.9:
        status = "STAGING_CANDIDATE"
        print("⚠️ Model meets staging criteria")
    else:
        status = "NEEDS_IMPROVEMENT"
        print("❌ Model needs improvement")
    
    return status

def promote_to_production(client, model_name, version, test_auc):
    """
    Simplified promotion without deprecated staging API
    """
    try:
        # Add production recommendation tags
        client.set_model_version_tag(
            name=model_name,
            version=version,
            key="recommended_for_production",
            value=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        client.set_model_version_tag(
            name=model_name,
            version=version,
            key="promotion_reason",
            value=f"Performance meets criteria - Test AUC: {test_auc:.4f}"
        )
        
        print(f"🚀 Model version {version} RECOMMENDED for production!")
        print(f"   Test AUC: {test_auc:.4f}")
        
    except Exception as e:
        print(f"Error adding promotion tags: {e}")

def final_training(config, best_params, X_train, y_train, X_val, y_val, categorical_columns):
    # Start MLflow run
    with mlflow.start_run(run_name="final_model_training") as run:
        # Train the final model with best parameters
        best_params.update({
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': 42
        })

        print("Training final LightGBM model with best parameters...")

        # Log hyperparameters
        mlflow.log_params(best_params)
        
        # Log data info
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_categorical_features", len(categorical_columns))

        # Create LightGBM datasets
        lgb_train = lgb.Dataset(
            X_train, 
            label=y_train,
            categorical_feature=categorical_columns
        )

        lgb_val = lgb.Dataset(
            X_val, 
            label=y_val,
            categorical_feature=categorical_columns,
            reference=lgb_train
        )

        # Train the final model
        final_model = lgb.train(
            best_params,
            lgb_train,
            num_boost_round=2000,
            valid_sets=[lgb_train, lgb_val],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(100)
            ]
        )

        # Log model metrics
        mlflow.log_metric("best_iteration", final_model.best_iteration)
        mlflow.log_metric("best_score", final_model.best_score['valid']['auc'])

        # Create model signature for Unity Catalog
        from mlflow.models.signature import infer_signature
        import numpy as np
        
        # Get a sample for signature inference
        sample_input = X_train.head(5)
        sample_predictions = final_model.predict(sample_input, num_iteration=final_model.best_iteration)
        
        # Infer signature
        signature = infer_signature(sample_input, sample_predictions)
        
        # Log the model WITH signature for Unity Catalog
        try:
            mlflow.lightgbm.log_model(
                final_model,
                artifact_path="lightgbm_model",  # Use artifact_path instead of name
                signature=signature,
                metadata={
                    "categorical_features": categorical_columns,
                    "feature_names": list(X_train.columns),
                    "model_type": "binary_classification",
                    "training_note": "Model trained with LightGBM native categorical support"
                }
            )
            print("✅ Model logged successfully with signature for Unity Catalog")
            
        except Exception as e:
            print(f"Error logging model with signature: {e}")
            # Fallback without signature
            mlflow.lightgbm.log_model(final_model, "lightgbm_model")
            print("✅ Model logged with minimal configuration")

        # save the final model locally as well
        model_path = config['data']['model_paths']['final_model']
        joblib.dump(final_model, model_path)
        print(f"Final model saved to {model_path}")

        # Log the model file as artifact
        mlflow.log_artifact(model_path, "local_model")

        import time
        time.sleep(5)

        # After model logging, verify it exists
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run.info.run_id)
        
        print("=== DEBUGGING ARTIFACTS ===")
        print(f"Total artifacts found: {len(artifacts)}")
        for artifact in artifacts:
            print(f"  - Path: '{artifact.path}', Is_dir: {artifact.is_dir}, Size: {getattr(artifact, 'file_size', 'N/A')}")
            # If it's a directory, also list its contents
            if artifact.is_dir:
                sub_artifacts = client.list_artifacts(run.info.run_id, artifact.path)
                for sub_artifact in sub_artifacts:
                    print(f"    - Subpath: '{sub_artifact.path}'")
        
        # More reliable verification: try to access the lightgbm_model directory directly
        print("=== ARTIFACT VERIFICATION ===")
        try:
            lgb_artifacts = client.list_artifacts(run.info.run_id, "lightgbm_model")
            if lgb_artifacts:
                print("✅ LightGBM model directory found and contains files:")
                for artifact in lgb_artifacts:
                    print(f"    - {artifact.path}")
                lightgbm_model_found = True
            else:
                print("❌ LightGBM model directory exists but is empty")
                lightgbm_model_found = False
        except Exception as e:
            print(f"❌ LightGBM model directory not found: {e}")
            lightgbm_model_found = False
        
        # Check for local model artifact
        local_model_found = any("local_model" in artifact.path for artifact in artifacts)
        if local_model_found:
            print("✅ Local model artifact verified in MLflow")
        else:
            print("❌ Local model artifact not found")

        # Also save categorical features info separately
        categorical_info = {
            "categorical_features": categorical_columns,
            "feature_names": list(X_train.columns),
            "feature_dtypes": {col: str(dtype) for col, dtype in X_train.dtypes.items()}
        }
        
        import json
        with open("categorical_features.json", "w") as f:
            json.dump(categorical_info, f, indent=2)
        mlflow.log_artifact("categorical_features.json")

        print(f"Training completed. Best iteration: {final_model.best_iteration}")
        print(f"MLflow run ID: {run.info.run_id}")

        return final_model, run.info.run_id

# evaluation
def evaluate_model_comprehensive(model, X_test, y_test, X_val, y_val, model_name="LightGBM"):
    """
    Comprehensive evaluation of the trained model with MLflow logging
    """
    print(f"=== {model_name} MODEL EVALUATION ===\n")
    
    # Predictions on test set
    y_test_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_test_pred = (y_test_pred_proba >= 0.5).astype(int)
    y_valid_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
    y_valid_pred = (y_valid_pred_proba >= 0.5).astype(int)
    
    # 1. Classification Reports
    print("\n--- TEST SET PERFORMANCE ---")
    test_report = classification_report(y_test, y_test_pred, target_names=['Good Loan', 'Risky Loan'], output_dict=True)
    print(classification_report(y_test, y_test_pred, target_names=['Good Loan', 'Risky Loan']))
    
    print("\n--- VALIDATION SET PERFORMANCE ---")
    val_report = classification_report(y_val, y_valid_pred, target_names=['Good Loan', 'Risky Loan'], output_dict=True)
    print(classification_report(y_val, y_valid_pred, target_names=['Good Loan', 'Risky Loan']))

    # 2. AUC-ROC Scores
    test_auc = roc_auc_score(y_test, y_test_pred_proba)
    valid_auc = roc_auc_score(y_val, y_valid_pred_proba)
    
    print(f"\n--- AUC-ROC SCORES ---")
    print(f"Test AUC-ROC: {test_auc:.4f}")
    print(f"Validation AUC-ROC: {valid_auc:.4f}")
    
    # 3. Average Precision Scores
    test_ap = average_precision_score(y_test, y_test_pred_proba)
    valid_ap = average_precision_score(y_val, y_valid_pred_proba)

    print(f"\n--- AVERAGE PRECISION SCORES ---")
    print(f"Test Average Precision: {test_ap:.4f}")
    print(f"Validation Average Precision: {valid_ap:.4f}")

    # Log metrics to MLflow
    if mlflow.active_run():
        # Test metrics
        mlflow.log_metric("test_auc_roc", test_auc)
        mlflow.log_metric("test_avg_precision", test_ap)
        mlflow.log_metric("test_precision_risky", test_report['Risky Loan']['precision'])
        mlflow.log_metric("test_recall_risky", test_report['Risky Loan']['recall'])
        mlflow.log_metric("test_f1_risky", test_report['Risky Loan']['f1-score'])
        mlflow.log_metric("test_accuracy", test_report['accuracy'])
        
        # Validation metrics
        mlflow.log_metric("val_auc_roc", valid_auc)
        mlflow.log_metric("val_avg_precision", valid_ap)
        mlflow.log_metric("val_precision_risky", val_report['Risky Loan']['precision'])
        mlflow.log_metric("val_recall_risky", val_report['Risky Loan']['recall'])
        mlflow.log_metric("val_f1_risky", val_report['Risky Loan']['f1-score'])
        mlflow.log_metric("val_accuracy", val_report['accuracy'])

        # Log classification reports as artifacts
        with open("test_classification_report.txt", "w") as f:
            f.write(classification_report(y_test, y_test_pred, target_names=['Good Loan', 'Risky Loan']))
        
        with open("val_classification_report.txt", "w") as f:
            f.write(classification_report(y_val, y_valid_pred, target_names=['Good Loan', 'Risky Loan']))
        
        mlflow.log_artifact("test_classification_report.txt")
        mlflow.log_artifact("val_classification_report.txt")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Final model training with MLflow tracking.')
    parser.add_argument('--config', required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Setup MLflow
    setup_mlflow(config)
    
    train_data, valid_data, test_data = load_loan_data(config)
    X_train, X_val, X_test, y_train, y_val, y_test, categorical_columns = prepare_lgb_data(
        train_data, valid_data, test_data
    )

    # Load best parameters from tuning
    best_params = load_best_params(config)
    
    # Train the final model using best parameters (MLflow run is started inside)
    final_model, run_id = final_training(config, best_params, X_train, y_train, X_val, y_val, categorical_columns)
    
    # Evaluate the final model (metrics logged to active MLflow run)
    evaluate_model_comprehensive(final_model, X_test, y_test, X_val, y_val)
    
    # Get performance metrics for model registration
    test_auc = roc_auc_score(y_test, final_model.predict(X_test, num_iteration=final_model.best_iteration))
    val_auc = roc_auc_score(y_val, final_model.predict(X_val, num_iteration=final_model.best_iteration))
    
    # Register model and potentially promote to production
    model_name = config.get('mlflow', {}).get('model_name', 'loan_risk_lightgbm')
    model_version, status = register_and_promote_model(model_name, run_id, test_auc, val_auc, config)
    
    if model_version:
        print(f"\n🎉 Model registration completed!")
        print(f"   Model Name: {model_name}")
        print(f"   Version: {model_version.version}")
        print(f"   Stage: {model_version.current_stage}")

    if status != "PRODUCTION_READY":
        print(f"\n⚠️ Model status: {status}")
        print("Please review the model performance and consider retraining or improving the model.")
        print("The workflow will stop here for non-production ready models.")
        exit(1)
        

# how to run
# py -3.12 src/model/final.py --config "config/config.yaml"