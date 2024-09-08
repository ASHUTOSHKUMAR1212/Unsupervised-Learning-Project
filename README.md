# Unsupervised-Learning-Project

Automobile Dataset Analysis and Unsupervised Learning
Part A: K-Means Clustering on Car Dataset
Project Overview

This project explores K-means clustering on a dataset of automobiles to segment cars into different categories based on their attributes. The goal is to understand K-means clustering and its application in the automotive industry for grouping cars based on similar features.
Data Description

The dataset includes:

    Multivalued Discrete Attributes:
        Cylinders, Model Year, Origin
    Continuous Attributes:
        Displacement, Horsepower, Weight, Acceleration, MPG (miles per gallon)
    Unique Identifier: Car Name

Project Objective

The objective is to apply K-means clustering to categorize the cars into different groups based on their features and analyze the results.
Steps and Tasks
1. Data Understanding & Exploration

    Load the dataset from Car name.csv and Car-Attributes.json files.
    Merge both dataframes into a single dataframe.
    Provide a 5-point summary for the numerical features and share insights.

2. Data Preparation & Analysis

    Check for missing and duplicate values and handle them appropriately.
    Visualize the data through pair plots and scatter plots to explore relationships between features such as wt, disp, and mpg, and observe how these features differ based on cyl.
    Check for any unexpected values in the dataset.

3. Clustering

    Apply K-Means clustering with 2 to 10 clusters.
    Visualize the elbow point to determine the optimal number of clusters.
    Train the K-Means model using the optimal number of clusters.
    Add cluster labels to the DataFrame and visualize clusters.
    Predict which cluster a new data point belongs to.

Part B: PCA and SVM Classification on Vehicle Silhouette Dataset
Project Overview

This project involves classifying vehicle silhouettes into one of three types (bus, van, car) based on geometric features extracted from the silhouettes. Dimensionality reduction using PCA is applied to improve the performance of a Support Vector Machine (SVM) classifier.
Data Description

The dataset consists of numeric features derived from the silhouettes of different vehicles:

    Vehicles include a double-decker bus, Chevrolet van, Saab 9000, and Opel Manta 400.
    The goal is to distinguish between these vehicles based on their geometric features.

Project Objective

The objective is to reduce the dimensions of the data using Principal Component Analysis (PCA) and apply an SVM classifier to predict vehicle types. Comparisons will be made to analyze model performance before and after dimensionality reduction.
Steps and Tasks
1. Data Understanding & Cleaning

    Load the dataset from vehicle.csv.
    Handle missing values and check for duplicate rows.
    Visualize the distribution of the target variable (class) using a pie chart.

2. Data Preparation

    Split the dataset into feature variables (X) and target (Y).
    Standardize the data.

3. Model Building

    Train a base SVM model and evaluate its performance on the train dataset.
    Apply PCA on the data to reduce dimensionality to 10 components.
    Visualize the cumulative variance explained and select components that capture 90% of the variance.
    Retrain the SVM on the reduced dataset and compare its performance to the base model.

4. Performance Improvement

    Tune the SVM hyperparameters to improve model performance after PCA.
    Report the best parameters and the relative improvement in model accuracy.

5. Theoretical Questions

    Explain the pre-requisites and assumptions of PCA.
    Discuss the advantages and limitations of PCA.

Tools and Libraries Used

    Python: Primary language for data analysis and modeling.
    Pandas: Data manipulation.
    NumPy: Numerical computations.
    Matplotlib/Seaborn: Data visualization.
    Scikit-learn: Machine learning algorithms, clustering, PCA, and SVM.

Conclusion

This project provides insights into the clustering of cars based on various attributes and the application of dimensionality reduction to improve classification performance. We have explored K-means clustering, PCA, and SVM classification, analyzing the performance of models before and after applying PCA.
