# the code is update to run on Docker

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import mlflow
import mlflow.lightgbm
import yaml
import json
from datetime import datetime
import logging
import warnings
import os
import lightgbm as lgb

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class ModelServer:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.model = None
        self.model_info = None
        self.categorical_features = None
        self.feature_names = None
        self.model_version = None
        self.load_model()
    
    def load_config(self, config_path):
        """Load configuration file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            return None
    
    def setup_mlflow(self):
        """Setup MLflow connection with proper experiment handling"""
        # Check if using Databricks
        if os.environ.get('MLFLOW_TRACKING_URI') == 'databricks':
            # Databricks setup
            tracking_uri = 'databricks'
            mlflow.set_tracking_uri(tracking_uri)
            
            # Set Unity Catalog registry URI
            mlflow.set_registry_uri("databricks-uc")
            logger.info(f"MLflow tracking URI set to: {tracking_uri}")
            logger.info("Using Unity Catalog Model Registry")
            
            # Verify Databricks environment variables
            databricks_host = os.environ.get('DATABRICKS_HOST')
            databricks_token = os.environ.get('DATABRICKS_TOKEN')
            
            if not databricks_host or not databricks_token:
                logger.error("DATABRICKS_HOST and DATABRICKS_TOKEN environment variables must be set")
                raise ValueError("Missing Databricks credentials")
            
            logger.info(f"Using Databricks workspace: {databricks_host}")
            
        else:
            # Local MLflow setup (existing Docker/local logic)
            if os.path.exists('/app') or os.environ.get('DOCKER_CONTAINER'):
                tracking_uri = 'file:///app/mlruns'
                os.environ['MLFLOW_TRACKING_URI'] = tracking_uri
            else:
                tracking_uri = 'file:./mlruns'
            
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI set to: {tracking_uri}")
        
        # ...existing experiment setup code...

    def get_latest_production_model(self):
        """Get the latest production-ready model from MLflow"""
        client = mlflow.tracking.MlflowClient()
        
        # Updated to handle Unity Catalog model name format
        model_name = self.config.get('mlflow', {}).get('model_name', 'loan_risk_lightgbm')
        
        # Check if we need to use Unity Catalog format
        if not model_name.count('.') == 2:  # Not in catalog.schema.model format
            catalog = self.config.get('mlflow', {}).get('catalog', 'workspace')
            schema = self.config.get('mlflow', {}).get('schema', 'default')
            model_name = f"{catalog}.{schema}.{model_name}"
        
        logger.info(f"Looking for model: {model_name}")
        
        try:
            # Get all versions of the model - fix the method call
            try:
                model_versions = client.search_model_versions(f"name='{model_name}'")
                # Convert to list if it's not already
                if hasattr(model_versions, '__iter__') and not isinstance(model_versions, (str, dict)):
                    model_versions = list(model_versions)
                else:
                    logger.error(f"Unexpected model_versions type: {type(model_versions)}")
                    model_versions = []
            except Exception as search_error:
                logger.error(f"Error searching for model versions: {search_error}")
                # Try alternative method for Unity Catalog
                try:
                    model_versions = client.get_latest_versions(model_name)
                    if hasattr(model_versions, '__iter__') and not isinstance(model_versions, (str, dict)):
                        model_versions = list(model_versions)
                    else:
                        model_versions = []
                except Exception as get_error:
                    logger.error(f"Error getting latest versions: {get_error}")
                    model_versions = []
            
            if not model_versions:
                logger.error(f"No model versions found for {model_name}")
                # Fallback to original model name format
                fallback_name = self.config.get('mlflow', {}).get('model_name', 'loan_risk_lightgbm')
                if fallback_name != model_name:
                    logger.info(f"Trying fallback model name: {fallback_name}")
                    try:
                        model_versions = client.search_model_versions(f"name='{fallback_name}'")
                        if hasattr(model_versions, '__iter__') and not isinstance(model_versions, (str, dict)):
                            model_versions = list(model_versions)
                        else:
                            model_versions = []
                        if model_versions:
                            model_name = fallback_name
                    except Exception as fallback_error:
                        logger.error(f"Fallback search failed: {fallback_error}")
                        model_versions = []
                
                if not model_versions:
                    return None, None
            
            logger.info(f"Found {len(model_versions)} model versions")
            
            # Find production-ready models
            production_ready_versions = []
            for version in model_versions:
                try:
                    # For Unity Catalog, we need to use get_model_version to get tags
                    tags = {}
                    if hasattr(version, 'name') and hasattr(version, 'version'):
                        try:
                            client = mlflow.tracking.MlflowClient()
                            detailed_version = client.get_model_version(version.name, version.version)
                            if hasattr(detailed_version, 'tags') and detailed_version.tags:
                                if isinstance(detailed_version.tags, dict):
                                    tags = detailed_version.tags
                                else:
                                    logger.warning(f"Unexpected detailed tags type: {type(detailed_version.tags)}")
                        except Exception as detail_error:
                            logger.warning(f"Error getting detailed model version for {version.name} v{version.version}: {detail_error}")
                    
                    logger.info(f"Version {getattr(version, 'version', 'unknown')} tags: {tags}")
                    
                    if tags.get('performance_status') == 'PRODUCTION_READY':
                        production_ready_versions.append(version)
                        
                except Exception as tag_error:
                    logger.warning(f"Error processing tags for version {getattr(version, 'version', 'unknown')}: {tag_error}")
                    continue
            
            if not production_ready_versions:
                logger.warning("No production-ready models found. Using latest version.")
                # Fallback to latest version
                try:
                    latest_version = max(model_versions, key=lambda x: int(x.version))
                    
                    # Use model registry URI instead of runs URI
                    model_uri = f"models:/{model_name}/{latest_version.version}"
                    logger.info(f"Using model registry URI: {model_uri}")
                    
                    return latest_version, model_uri
                except Exception as latest_error:
                    logger.error(f"Error getting latest version: {latest_error}")
                    return None, None
            
            # Get the latest production-ready version
            try:
                latest_prod_version = max(production_ready_versions, key=lambda x: int(x.version))
                
                # Use model registry URI for production version
                model_uri = f"models:/{model_name}/{latest_prod_version.version}"
                logger.info(f"Using model registry URI: {model_uri}")

                logger.info(f"Loading production model version {latest_prod_version.version}")
                return latest_prod_version, model_uri
            except Exception as prod_error:
                logger.error(f"Error getting production version: {prod_error}")
                return None, None
                
        except Exception as e:
            logger.error(f"Error getting production model: {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None, None
    
    def load_model(self):
        """Load the latest production model"""
        try:
            self.setup_mlflow()
            
            # Get production model
            model_version, model_uri = self.get_latest_production_model()
            
            if model_uri is None:
                logger.error("Failed to get model URI")
                return False
            
            # Check if we're using Databricks
            is_databricks = os.environ.get('MLFLOW_TRACKING_URI') == 'databricks'
            
            if is_databricks:
                # For Databricks, use the model registry URI directly
                logger.info(f"Using Databricks model registry URI: {model_uri}")
                # Don't try to convert to local paths for Databricks
                docker_model_uri = model_uri
            else:
                # Original Docker/local logic for non-Databricks setups
                if os.path.exists('/app') or os.environ.get('DOCKER_CONTAINER'):
                    logger.info("Running in Docker container - converting model URI to Docker paths")
                    
                    # For Docker, always use direct file paths instead of model registry URIs
                    if 'models:/' in model_uri or model_version:
                        # Build the Docker path directly from run_id
                        run_id = model_version.run_id
                        
                        # Search through all available experiments to find the run_id
                        mlruns_dir = "/app/mlruns"
                        found_run_path = None
                        
                        if os.path.exists(mlruns_dir):
                            for exp_dir in os.listdir(mlruns_dir):
                                exp_path = os.path.join(mlruns_dir, exp_dir)
                                if os.path.isdir(exp_path) and exp_dir not in ['.trash', 'models']:
                                    run_path = os.path.join(exp_path, run_id)
                                    if os.path.exists(run_path):
                                        found_run_path = run_path
                                        logger.info(f"Found run {run_id} in experiment {exp_dir}")
                                        break
                        
                        if not found_run_path:
                            logger.error(f"Could not find run_id {run_id} in any experiment")
                            return False
                        
                        # Try different possible artifact paths
                        possible_paths = [
                            f"{found_run_path}/artifacts/model",
                            f"{found_run_path}/artifacts/local_model", 
                            f"{found_run_path}/artifacts/lightgbm-model",
                            f"{found_run_path}/artifacts",
                        ]
                    
                        docker_model_uri = None
                        for path in possible_paths:
                            if os.path.exists(path):
                                logger.info(f"Checking path: {path}")
                                # Check if it contains model files
                                if os.path.exists(os.path.join(path, "MLmodel")):
                                    docker_model_uri = f"file://{path}"
                                    logger.info(f"Found MLmodel file at: {docker_model_uri}")
                                    break
                                elif any(f.endswith('.txt') or f.endswith('.pkl') or f.endswith('.json') for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))):
                                    # Check if this is the local_model directory with actual model files
                                    if os.path.basename(path) == 'local_model':
                                        # For local_model directory, we need to load the model differently
                                        # Try to find the actual model files
                                        model_files = [f for f in os.listdir(path) if f.endswith('.txt') or f.endswith('.pkl')]
                                        if model_files:
                                            logger.info(f"Found model files in local_model: {model_files}")
                                            # Use a custom loading approach for local_model
                                            docker_model_uri = f"local_model:{path}"
                                            logger.info(f"Using local_model path: {docker_model_uri}")
                                            break
                                    else:
                                        logger.info(f"Path exists but no MLmodel file found: {path}")
                                        if os.path.isdir(path):
                                            logger.info(f"Contents: {os.listdir(path)}")
                                else:
                                    logger.info(f"Path exists but no model files found: {path}")
                                    if os.path.isdir(path):
                                        logger.info(f"Contents: {os.listdir(path)}")
                                        
                        if docker_model_uri is None:
                            logger.error(f"Could not find model artifacts in Docker filesystem for run_id: {run_id}")
                            return False
                else:
                    docker_model_uri = model_uri
            
            # Load the model
            if docker_model_uri and docker_model_uri.startswith('local_model:'):
                # Custom loading for local_model directory
                local_model_path = docker_model_uri.replace('local_model:', '')
                
                # Try to find the model file - check for different formats
                model_files = [f for f in os.listdir(local_model_path) if f.endswith('.pkl') or f.endswith('.txt')]
                if model_files:
                    model_file_path = os.path.join(local_model_path, model_files[0])
                    logger.info(f"Loading model from: {model_file_path}")
                    
                    # Determine loading method based on file extension
                    if model_file_path.endswith('.pkl'):
                        # Load pickled model
                        import pickle
                        with open(model_file_path, 'rb') as f:
                            self.model = pickle.load(f)
                        logger.info("Loaded pickled model successfully")
                    elif model_file_path.endswith('.txt'):
                        # Load LightGBM text model
                        self.model = lgb.Booster(model_file=model_file_path)
                        logger.info("Loaded LightGBM text model successfully")
                else:
                    logger.error("No model file (.pkl or .txt) found in local_model directory")
                    return False
            else:
                # Use MLflow to load the model (works for both Databricks and local)
                logger.info(f"Loading model using MLflow from URI: {docker_model_uri}")
                self.model = mlflow.lightgbm.load_model(docker_model_uri)
                logger.info("Model loaded successfully using MLflow")
            
            self.model_version = model_version
            
            # Load categorical features info
            try:
                if is_databricks:
                    # For Databricks, use MLflow client to download artifacts
                    logger.info("Loading categorical features from Databricks")
                    client = mlflow.tracking.MlflowClient()
                    artifact_path = client.download_artifacts(
                        model_version.run_id, 
                        "categorical_features.json"
                    )
                    
                    with open(artifact_path, 'r') as f:
                        categorical_info = json.load(f)
                        self.categorical_features = categorical_info.get('categorical_features', [])
                        self.feature_names = categorical_info.get('feature_names', [])
                        logger.info(f"Loaded categorical features from Databricks: {len(self.categorical_features)} features")
                else:
                    # Original Docker/local logic for categorical features
                    # For Docker environment, try direct file path first
                    if os.path.exists('/app') or os.environ.get('DOCKER_CONTAINER'):
                        # Try to find categorical_features.json in the artifacts directory
                        run_id = model_version.run_id
                        
                        # Get experiment ID
                        experiment_id = None
                        try:
                            client = mlflow.tracking.MlflowClient()
                            run = client.get_run(run_id)
                            experiment_id = run.info.experiment_id
                        except Exception as e:
                            logger.warning(f"Could not get experiment ID: {e}")
                        
                        # Try different possible paths for categorical_features.json
                        possible_paths = [
                            f"/app/mlruns/{run_id}/artifacts/categorical_features.json",
                            f"/app/mlruns/{experiment_id}/{run_id}/artifacts/categorical_features.json" if experiment_id else None,
                            f"/app/mlruns/{run_id}/artifacts/local_model/categorical_features.json",
                            f"/app/mlruns/{experiment_id}/{run_id}/artifacts/local_model/categorical_features.json" if experiment_id else None,
                        ]
                        
                        # Remove None values
                        possible_paths = [path for path in possible_paths if path is not None]
                        
                        categorical_file_path = None
                        for path in possible_paths:
                            if os.path.exists(path):
                                categorical_file_path = path
                                logger.info(f"Found categorical features file at: {path}")
                                break
                        
                        if categorical_file_path:
                            with open(categorical_file_path, 'r') as f:
                                categorical_info = json.load(f)
                                self.categorical_features = categorical_info.get('categorical_features', [])
                                self.feature_names = categorical_info.get('feature_names', [])
                                logger.info(f"Loaded categorical features: {len(self.categorical_features)} features")
                        else:
                            # Fallback to MLflow client method
                            logger.info("Trying MLflow client download_artifacts as fallback")
                            client = mlflow.tracking.MlflowClient()
                            artifact_path = client.download_artifacts(
                                model_version.run_id, 
                                "categorical_features.json"
                            )
                            
                            with open(artifact_path, 'r') as f:
                                categorical_info = json.load(f)
                                self.categorical_features = categorical_info.get('categorical_features', [])
                                self.feature_names = categorical_info.get('feature_names', [])
                    else:
                        # Local development - use MLflow client
                        client = mlflow.tracking.MlflowClient()
                        artifact_path = client.download_artifacts(
                            model_version.run_id, 
                            "categorical_features.json"
                        )
                        
                        with open(artifact_path, 'r') as f:
                            categorical_info = json.load(f)
                            self.categorical_features = categorical_info.get('categorical_features', [])
                            self.feature_names = categorical_info.get('feature_names', [])
                            
            except Exception as e:
                logger.warning(f"Could not load categorical features info: {e}")
                self.categorical_features = []
                self.feature_names = []
            
            # Store model info
            if model_version:
                # Get tags using get_model_version for Unity Catalog
                tags = {}
                try:
                    if hasattr(model_version, 'name') and hasattr(model_version, 'version'):
                        client = mlflow.tracking.MlflowClient()
                        detailed_version = client.get_model_version(model_version.name, model_version.version)
                        if hasattr(detailed_version, 'tags') and detailed_version.tags:
                            if isinstance(detailed_version.tags, dict):
                                tags = detailed_version.tags
                except Exception as e:
                    logger.warning(f"Could not get model version tags: {e}")
                
                self.model_info = {
                    'model_name': model_version.name,
                    'version': model_version.version,
                    'run_id': model_version.run_id,
                    'test_auc': tags.get('test_auc', 'N/A'),
                    'val_auc': tags.get('val_auc', 'N/A'),
                    'performance_status': tags.get('performance_status', 'N/A'),
                    'training_date': tags.get('training_date', 'N/A'),
                    'loaded_at': datetime.now().isoformat()
                }
            else:
                self.model_info = {
                    'model_name': 'unknown',
                    'version': 'unknown',
                    'run_id': 'unknown',
                    'test_auc': 'N/A',
                    'val_auc': 'N/A',
                    'performance_status': 'N/A',
                    'training_date': 'N/A',
                    'loaded_at': datetime.now().isoformat()
                }
            
            logger.info(f"Model loaded successfully: {self.model_info}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def preprocess_input(self, data):
        """Preprocess input data to match training format"""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
        
        # Ensure all expected features are present
        if self.feature_names:
            missing_features = set(self.feature_names) - set(df.columns)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                # Add missing features with default values
                for feature in missing_features:
                    df[feature] = None
        
        # Convert categorical features to category dtype
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Reorder columns to match training data if feature names available
        if self.feature_names:
            df = df.reindex(columns=self.feature_names, fill_value=None)
        
        return df
    
    def predict_single(self, input_data):
        """Make prediction for single instance"""
        try:
            df = self.preprocess_input(input_data)
            
            # Get prediction probability
            prediction_proba = self.model.predict(df)[0]
            prediction = 1 if prediction_proba >= 0.5 else 0
            
            return {
                'prediction': int(prediction),
                'probability': float(prediction_proba),
                'risk_level': 'High Risk' if prediction == 1 else 'Low Risk',
                'confidence': float(abs(prediction_proba - 0.5) + 0.5)
            }
            
        except Exception as e:
            logger.error(f"Error in single prediction: {e}")
            raise e
    
    def predict_batch(self, input_data):
        """Make predictions for batch of instances"""
        try:
            df = self.preprocess_input(input_data)
            
            # Get prediction probabilities
            prediction_probas = self.model.predict(df)
            predictions = (prediction_probas >= 0.5).astype(int)
            
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, prediction_probas)):
                results.append({
                    'id': i,
                    'prediction': int(pred),
                    'probability': float(prob),
                    'risk_level': 'High Risk' if pred == 1 else 'Low Risk',
                    'confidence': float(abs(prob - 0.5) + 0.5)
                })
            
            return {
                'predictions': results,
                'summary': {
                    'total_predictions': len(results),
                    'high_risk_count': sum(1 for r in results if r['prediction'] == 1),
                    'low_risk_count': sum(1 for r in results if r['prediction'] == 0),
                    'average_probability': float(np.mean(prediction_probas))
                }
            }
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            raise e

# Initialize model server
model_server = ModelServer('config/config.yaml')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        if model_server.model is None:
            return jsonify({
                'status': 'unhealthy',
                'message': 'Model not loaded',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        # Test prediction with dummy data to ensure model works
        test_data = {col: 0 for col in model_server.feature_names} if model_server.feature_names else {'test': 0}
        try:
            _ = model_server.predict_single(test_data)
            model_status = 'healthy'
            model_message = 'Model is functioning correctly'
        except Exception as e:
            model_status = 'degraded'
            model_message = f'Model prediction test failed: {str(e)}'
        
        return jsonify({
            'status': model_status,
            'message': model_message,
            'model_info': model_server.model_info,
            'features': {
                'total_features': len(model_server.feature_names) if model_server.feature_names else 0,
                'categorical_features': len(model_server.categorical_features)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Health check failed: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict', methods=['POST'])
def predict_single():
    """Single prediction endpoint"""
    try:
        if model_server.model is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        data = request.json
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        result = model_server.predict_single(data)
        
        return jsonify({
            'success': True,
            'result': result,
            'model_version': model_server.model_info['version'] if model_server.model_info else 'unknown',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    try:
        if model_server.model is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        data = request.json
        if not data or 'data' not in data:
            return jsonify({'error': 'No input data provided. Expected format: {"data": [...]}'}), 400
        
        input_data = data['data']
        if not isinstance(input_data, list):
            return jsonify({'error': 'Input data must be a list of records'}), 400
        
        if len(input_data) > 1000:  # Limit batch size
            return jsonify({'error': 'Batch size too large. Maximum 1000 records allowed'}), 400
        
        result = model_server.predict_batch(input_data)
        
        return jsonify({
            'success': True,
            'result': result,
            'model_version': model_server.model_info['version'] if model_server.model_info else 'unknown',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        if model_server.model is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        return jsonify({
            'model_info': model_server.model_info,
            'features': {
                'total_features': len(model_server.feature_names) if model_server.feature_names else 0,
                'feature_names': model_server.feature_names,
                'categorical_features': model_server.categorical_features
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/model/reload', methods=['POST'])
def reload_model():
    """Reload the latest model"""
    try:
        success = model_server.load_model()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Model reloaded successfully',
                'model_info': model_server.model_info,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to reload model',
                'timestamp': datetime.now().isoformat()
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/', methods=['GET'])
def home():
    """API documentation"""
    return jsonify({
        'message': 'Loan Risk Prediction API',
        'version': '1.0',
        'endpoints': {
            'health': 'GET /health - Check API and model health',
            'predict': 'POST /predict - Single prediction',
            'predict_batch': 'POST /predict/batch - Batch predictions',
            'model_info': 'GET /model/info - Get model information',
            'reload_model': 'POST /model/reload - Reload latest model'
        },
        'model_status': 'loaded' if model_server.model is not None else 'not_loaded',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)