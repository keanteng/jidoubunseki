# ML Assessment

This repository contains the code and documentation for the ML assessment project. The goal of this project is to build an automated machine learning system to predict the risk of loan applications

The project is coded on Python version 3.12.

## Repository Structure

```bash

```

## Using This Repo

### Local Development
```powershell
docker build -t loan-risk-api:latest .
docker-compose up --build
```

### Building and Pushing to Docker Hub
```powershell
# Build with version tag
docker build -t keantengblog/loan-risk-api:v1.0.0 -t keantengblog/loan-risk-api:latest .

# Push to Docker Hub
docker push keantengblog/loan-risk-api:v1.0.0
docker push keantengblog/loan-risk-api:latest
```

### Pull and Run from Docker Hub
```powershell
docker pull keantengblog/loan-risk-api:latest
docker run -d -p 5000:5000 --name loan-risk-api keantengblog/loan-risk-api:latest

# py -3.12 -m venv venv
# source "/C/Users/Khor Kean Teng/Downloads/Git/ml-assessment/venv/Scripts/activate"
# pip install -r requirements-2.txt
# bash bin/pipeline.sh
# deactivate


git lfs install
git lfs track "*.pkl" "*.h5" "*.model" "*.bin" ".csv"
git add .gitattributes
git commit -m "Add LFS tracking"
git push origin main