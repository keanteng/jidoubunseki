---
author: "Khor Kean Teng"
title: "Part 2 - Automated ML System"
date: 2025-08-05
---

# Part 2 - Automated ML System

The diagram below presents the proposed architecture for an automated machine learning system:

![alt text](image-1.png)

## Architecture Overview

The architecture make use of GitHub Actions, Github repository as well as Docker to automate the machine learning workflow.

The workflow can be either triggered periodically, manually (through GitHub Action or pushing changes to the repository).

Once the workflow is triggered it will first perform data validation using Great Expectation to ensure the integrity of the data. Three files are expected to be present, the `loan`, `payment` and the `clarity_underwriting_variable`. Then the dataset will be joined according to the identifiers.

Then, the data will be preprocessed using prewritten python scripts to perform tasks such as imputations, feature engineering and removal of unnecessary features such as identifiers. 

Then, the processed data will be splitted into training, testing and validation dataset in 70, 15 and 15 ratio respectively. The splitting is done using time based strategy to ensure that the model is trained on past data and tested on future data, avoiding data leakage. 

The modeling starts with hyperparameter tuning using Optuna, monitored by MLFlow this ensure that each trigger of the workflow will retain the best model and hyperparameters and the relevant metrics. In each trigger, the best parameters will be passed to train a final model, which will be logged into MLFlow as well.

Then, a promotion criteria will be performed to determine whether the model should be promoted to production. The promotion system will check the performance of the model against certain criteria like minimum recall and precision or other metrics such as 2% better than the previous model in terms of F1 score. In our case, we just make sure the precision and recall is above 0.72 as part of demonstration.

Then the promoted model will be deployed using Flask, containerized on Docker so that it can be accessed through API for single or batch prediction. To make sure the model works after deployed, a test script will also be triggered to ensure the model can be accessed and the API is working as expected.

Then the Docker image will be pushed to the Docker Hub repository where we can deploy the services by pulling the images.