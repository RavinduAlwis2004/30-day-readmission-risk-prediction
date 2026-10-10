# 30-Day Hospital Readmission Risk Prediction

## Overview

This project explores how machine learning can support the identification of hospital encounters associated with readmission within 30 days. It uses the Diabetes 130-US Hospitals dataset and compares multiple classification models.

The project was developed as part of the IT3091 Machine Learning module at SLIIT.

## Objectives

- Formulate a healthcare readmission problem as a supervised classification task.
- Explore and preprocess patient encounter data.
- Compare a baseline model with Logistic Regression, Decision Tree, Random Forest and Linear SVM.
- Evaluate models using precision, recall, F1-score, ROC-AUC and confusion matrices.
- Build a Streamlit dashboard to demonstrate predictions.

## Dataset

The dataset contains 101,766 hospital encounters and 50 original columns. The target was converted into a binary outcome:

- `1`: Readmission within 30 days (`<30`)
- `0`: No readmission within 30 days (`>30` or `NO`)

Dataset source: [UCI Machine Learning Repository — Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes%2B130-us+hospitals%2Bfor+years+1999-2008)

## Models Compared

- Dummy Classifier (baseline)
- Logistic Regression
- Decision Tree
- Random Forest
- Linear Support Vector Machine

## Selected Model

The 15-feature Random Forest model was selected based on its overall performance across the tested models.

Held-out test results:

- Accuracy: 65.32%
- Precision: 17.20%
- Recall: 53.08%
- F1-score: 25.98%
- ROC-AUC: 63.45%

These results indicate moderate predictive performance. Accuracy alone is not sufficient because the positive class represents approximately 11.16% of encounters.

## Dashboard

A Streamlit prototype accepts selected patient and encounter features and displays a predicted class and model probability.

## Technologies

Python, pandas, NumPy, scikit-learn, Matplotlib, Jupyter Notebook and Streamlit.

## Limitations

This is an academic prototype using historical data. Its predictions have not been clinically validated, and the displayed probabilities have not been clinically calibrated. It must not be used as an autonomous clinical decision-making tool.

## Reproducibility

Install the required packages, run the notebook to reproduce the modelling workflow, and run the Streamlit application using the instructions provided in the project files.

## Academic Context

Developed as a group assignment for IT3091 Machine Learning, SLIIT. Contributions should be credited accurately to the relevant team members.
