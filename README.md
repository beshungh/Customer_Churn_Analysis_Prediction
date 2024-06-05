# Bank Customer Churn Prediction

### Project Overview
 The aim of this project is to predict customer churn for a bank using data analysis and machine learning techniques. This demonstrates the entire process, including data 
 preprocessing, exploratory data analysis, feature engineering, model building, evaluation, and recommendations.

## *Table of Contents*

- *[Introduction](#project-summary)*
- *[Data Analysis](#data-source)*
- *[Data Preprocessing](#tools-used)*
- *[Feature Engineering](#importing-csv-files-into-postgresql-using-python-script)*
- *[Model Building](#setbacks-of-the-python-script)*
- *[Findings And Recommendations](#entity-relationship-diagram)*
- *[Conclusion](#creating-database-tables)*

### Context
Customer churn is a critical metric for banks, representing the degree of customer inactivity or disengagement over a given period. It can manifest in various ways within the data, such as the recency of account actions or changes in account balance. Understanding the factors that contribute to customer churn allows banks to proactively address issues and improve customer retention strategies.

### Aim
The primary objectives of this study are to:

- Identify and visualize the key factors that contribute to customer churn.
- Build a robust prediction model to classify whether a customer is likely to churn.
- Choose a model that can provide probabilities of churn, helping customer service teams target efforts more effectively.

### Project Timeline
The project will be executed in the following stages:

1. **Data Analysis**: Explore and understand the dataset, identifying trends and patterns related to customer churn.
2. **Feature Engineering**: Create and select relevant features that will improve the model's predictive power.
3. **Model Building using ANN**: Develop an Artificial Neural Network (ANN) model to predict customer churn.
4. **Model Building and Prediction using H2O AutoML**: Utilize H2O AutoML to build and compare multiple machine learning models, selecting the best-performing model for deployment.

### Model Building and Prediction

#### The Sequential Model
A Sequential model is a type of neural network model that is appropriate for a simple stack of layers, where each layer has exactly one input tensor and one output tensor. This model is straightforward to implement and is ideal for linear topologies. However, it is not suitable for complex models with multiple inputs or outputs, layer sharing, or non-linear topologies (such as residual connections or multi-branch models).

#### H2O AutoML
H2O is an open-source, distributed, in-memory machine learning platform that scales linearly. H2O supports a wide range of statistical and machine learning algorithms, including gradient boosted machines, generalized linear models, and deep learning. H2O AutoML automates the process of training and tuning a large selection of models, providing an easy-to-use interface for quickly finding the best model for your data.

### Key Terms Explained

- **Churn**: The process of customers stopping their business with a company. In banking, this might mean closing accounts or significantly reducing their usage.
- **Artificial Neural Network (ANN)**: A type of machine learning model inspired by the human brain, capable of recognizing complex patterns and relationships within data.
- **Feature Engineering**: The process of creating new features or modifying existing ones to improve the performance of a machine learning model.
- **H2O AutoML**: A platform that automates the process of building and tuning machine learning models, making it accessible even for those with limited machine learning expertise.
