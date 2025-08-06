# ML Assessment

This repository contains the code and documentation for the ML assessment project. The goal of this project is to build an automated machine learning system to predict the risk of loan applications

The project is coded on Python version 3.12, tested on `venv` with Python version 3.12 and deployed using GitHub Actions and Docker.

The architectural details can be found by reading the documents in the `docs` folder.

*Tech Used: Python, Docker, GitHub Actions, MLflow, Databricks and Optuna.*

## Repository Structure

To structure can be created on `bash` by typing: `cmd //c tree //a`

```bash
C:.
+---.github  # the workflow fo GitHub Actions
|   \---workflows
+---archive # the archived files
+---assessment # files from the assessment
|   +---data
|   |   \---data
|   \---dictionaries
|       \---dictionaries
+---bin # bash scripts for running the pipeline
+---config # pipeline configuaration variables
+---data # pipeline artifacts
|   +---processed
|   \---raw
+---docs # documentation files
+---logs # placeholder for logs
+---mlruns # artifacts from MLflow
+---notebooks # pipeline all in one notebook
+---public # pipeline architecture diagrams
|   \---architecture
+---src # pipeline python scripts
|   +---data
|   +---model
|   +---preprocessing
|   \---utils
\---test # api testing scripts
```

## Using This Repo

Usage of this repository required Docker, Git and Python 3.12 to be installed on your machine. The repository can be cloned using the following command:

```bash
git clone https://github.com/keanteng/jidoubunseki
```

The repository can be verified by checking the following sections:

- Testing for Pipeline Using `venv`
- Testing for API services using docker
- Testing Workflow using GitHub Actions on Fresh Repository

## Testing for Pipeline Using `venv`

First create a virtual `venv` on Python and run the following code. Note that Databricks information can be obtained from Databricks account, the host is the URL of the workspace and the token can be generated from the user settings by going to `Settings > Developer > Access Token`.

```bash
py -3.12 -m venv venv
source "/C/Users/Khor Kean Teng/Downloads/Git/ml-assessment/venv/Scripts/activate"

# once the virtual environment is activated, install the required packages
pip install -r requirements-2.txt

# run the pipeline
# configure the env variables first
export DATABRICKS_HOST=<databricks_host>
export DATABRICKS_TOKEN=<databricks_token>
export MLFLOW_TRACKING_URI="databricks"
bash bin/pipeline.sh

# after the pipeline is run, you can deactivate the virtual environment
deactivate
```

## Testing for API services using Docker

### Local Development

You can build the Docker image by after cloning the repository:

```powershell
docker build -t loan-risk-api:latest .
docker-compose up --build
```

### Building and Pushing to Docker Hub

To push the build image to Docker Hub, the following commands can be modified such as version code and repository name:

```powershell
# Build with version tag
docker build -t keantengblog/loan-risk-api:v1.0.0 -t keantengblog/loan-risk-api:latest .

# Push to Docker Hub
docker push keantengblog/loan-risk-api:v1.0.0
docker push keantengblog/loan-risk-api:latest
```

### Pull and Run from Docker Hub

To check and verify the Docker image, you can pull the latest image from Docker Hub and run it locally:

```powershell
docker pull keantengblog/loan-risk-api:latest
docker run -d -p 5000:5000 --name loan-risk-api keantengblog/loan-risk-api:latest
```

Verifying all the API services can be done by running the test scripts:

```bash
# test for single prediction and batch prediction
# the script will check port 5000, please change it if you are using a different port or if the service is live on a different host
py -3.12 test/api-test.py
```

## Testing Workflow using GitHub Actions on Fresh Repository

First of all, create the following credentials in the GitHub repository by going to `Settings > Secrets and variables > Actions`:

```bash
DATABRICKS_HOST=<databricks_host>
DATABRICKS_TOKEN=<databricks_token> # this is the token make sure got read, write and delete permissions
MLFLOW_TRACKING_URI="databricks"
DOCKER_USERNAME=<docker_username>
DOCKER_PASSWORD=<docker_password> # this is the token make sure got read, write and delete permissions
```

Then run the following code:

```bash
# init git
git init

# track large files using Git LFS
git lfs install
git lfs track "*.pkl" "*.h5" "*.model" "*.bin" ".csv"
git add .gitattributes
git commit -m "Add LFS tracking"

# add all files
git add .

# commit the changes
git commit -m "Initial commit"

# add remote repository
git remote add origin <remote_repository_url>

# push the changes to the remote repository
git branch -M main

# push to remote repository
git push origin main
```