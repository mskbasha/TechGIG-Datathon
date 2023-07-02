# TechGIG-Datathon
# Healthcare Professional Classification and Taxonomy Prediction

This repository contains the code and resources for a project aimed at predicting whether a user in ad server logs belongs to the category of Healthcare Professionals (HCP) and determining their specialization or taxonomy if they are an HCP.

## Problem Description

The objective of this project is to build a robust model that accurately predicts whether a user is an HCP and their specialization based on ad server logs. The input data includes information such as user behavior, browser details, IP addresses, geographic locations, search patterns, and site URLs.

## Solution Approach

The project follows the following approach to solve the problem:

1. Exploratory Data Analysis (EDA): Perform data exploration and gain insights into the dataset. This includes analyzing data distributions, identifying patterns, and handling missing values or outliers.

2. Data Preprocessing: Convert the keywords column into numerical values using the GLOVE model, which represents words as high-dimensional vectors. Apply target encoding to handle categorical variables and prevent overfitting.

3. HCP Classification: Use random forests, an ensemble learning method, to predict whether a user is an HCP or not. Train the model with a suitable number of estimators to improve prediction accuracy.

4. Taxonomy Prediction: Predict the taxonomy or specialization for HCP users using a sequential approach. Use multiple models, each trained to predict a specific taxonomy label. Start by predicting the most frequent label and then pass the remaining data to subsequent models for further prediction until all taxonomy labels are covered.

5. Evaluation and Scoring: Evaluate the models' performance using appropriate metrics such as accuracy, precision, recall, and F1-score. Generate the output file with the required parameters for scoring the solution.



## Dependencies

- Python (version 3.9)
- Pandas 
- NumPy 
- Scikit-learn 
- GLOVE 


## Results

- f1_score max of 99.8671
- max of 81.12 for Taxonomy with most frequent label


