#!/bin/ba
echo "🚀Starting ML Pipeline..."

# Create directories if they don't exist
mkdir -p data/raw data/processed models config

# Run ingestion 
echo "📥 Step 0: Data Ingestion..."
python src/ingestion/ingest.py

# Run pipeline steps
echo "📊 Step 1: Data Merging..."
python src/data/merge.py --config config/config.yaml

echo "🧹 Step 2: Data Preprocessing..."
python src/preprocessing/preprocess.py --config config/config.yaml

echo "✂️ Step 3: Data Splitting..."
python src/model/split.py --config config/config.yaml

echo "🎯 Step 4: Hyperparameter Tuning..."
python src/model/tune.py --config config/config.yaml

echo "🏋️ Step 5: Final Model Training..."
python src/model/final.py --config config/config.yaml

echo "✅ Pipeline completed successfully!"