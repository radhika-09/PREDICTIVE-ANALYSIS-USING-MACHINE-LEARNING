# PREDICTIVE-ANALYSIS-USING-MACHINE-LEARNING
Project Title: Iris Flower Classification
Company: Codtech IT Solutions
Name: Radhika
Intern ID: CT04DF1997
Domain: Data Analytics
Duration: 4 Weeks
Mentor: Neela Santosh

🌸 Iris Flower Classification – CODTECH Internship Task 2
This repository contains the implementation of Task 2 from the CODTECH Internship Program, where the goal is to build machine learning models to perform predictive analysis on a dataset. The selected dataset for this task is the classic Iris flower dataset, and two different classification models have been applied: Random Forest and K-Nearest Neighbors (KNN). The objective is to train these models, perform feature selection, evaluate them, and compare their accuracy.

The Iris dataset is one of the most popular datasets used in pattern recognition. It contains 150 records under five attributes. Each record represents a type of Iris flower with four measurable features and one target class (species). The features include the lengths and widths of the petals and sepals. The species belong to three categories: Iris Setosa, Iris Versicolor, and Iris Virginica.

🔍 Objective
The main goal of this project is to:

Load and understand the structure of the Iris dataset.

Apply feature selection to reduce dimensionality and keep the most relevant attributes.

Train two machine learning models: Random Forest and KNN.

Evaluate the models using standard classification metrics.

Compare the performance of both models and determine which performs better on the dataset.

📌 Technologies & Tools Used
Python

Jupyter Notebook

Pandas

Seaborn and Matplotlib (for data visualization)

Scikit-learn (for preprocessing, feature selection, model training, and evaluation)

📊 Approach & Methodology
Data Loading & Preprocessing:
The dataset was loaded using Seaborn’s built-in function. The species column was encoded using LabelEncoder to convert categorical labels into numeric format.

Feature Selection:
To enhance model efficiency and performance, SelectKBest with ANOVA F-value was used to choose the top 3 most significant features. This also helps reduce overfitting and speeds up computation.

Data Scaling:
Since KNN is a distance-based algorithm, StandardScaler was applied to normalize the selected features. This ensures that all features contribute equally to distance measurements.

Model Training:

Random Forest Classifier: An ensemble learning model based on decision trees. It performs well with minimal tuning.

K-Nearest Neighbors (KNN): A simple but powerful algorithm that classifies data points based on their closest neighbors in the feature space.

Evaluation Metrics:
Both models were evaluated using:

Accuracy Score

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

📈 Results
The Random Forest and KNN model achieved approximately 100% accuracy, depending on the train-test split. This showcases the strength of ensemble models, although KNN also performed well with proper scaling.

