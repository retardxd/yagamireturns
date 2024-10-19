# Importing necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to the Adult dataset
adult_dataset_path = "/content/adult.csv"

# Function to load the Adult dataset
def load_adult_data(adult_path=adult_dataset_path):
    # Join the given path and return the DataFrame after reading the CSV file
    csv_path = os.path.join(adult_path)
    return pd.read_csv(csv_path)

# Load the adult dataset and assign it to a variable
df = load_adult_data()

# Display the top 3 rows of the dataset
df.head(3)

# Print the shape of the DataFrame to see the number of rows and columns
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Print the names of the features (columns)
print("\nFeatures:\n", df.columns.tolist())

# Check for missing values in the dataset
print("\nMissing values:", df.isnull().sum().values.sum())

# Check the number of unique values in each column
print("\nUnique values:\n", df.nunique())

# Display basic information about the dataset
df.info()

# Display descriptive statistics for the dataset
df.describe()

# Check for missing values represented as '?'
df_check_missing_workclass = (df['workclass'] == '?').sum()  # Count missing values in 'workclass'
df_check_missing_occupation = (df['occupation'] == '?').sum()  # Count missing values in 'occupation'

# Create a DataFrame to check for all missing values represented as '?'
df_missing = (df == '?').sum()

# Calculate the percentage of missing values for each column
percent_missing = (df == '?').sum() * 100 / len(df)

# Apply a lambda function to check the count of non-'?' values in each row
df.apply(lambda x: x != '?', axis=1).sum()

# Select all categorical variables from the DataFrame
df_categorical = df.select_dtypes(include=['object'])

# Check if any other columns contain the '?' value
df_categorical.apply(lambda x: x == '?', axis=1).sum()

# Drop rows where 'occupation' or 'native.country' has the '?' value
df = df[df['occupation'] != '?']
df = df[df['native-country'] != '?']
df.info()

# Importing the preprocessing module for encoding categorical variables
from sklearn import preprocessing

# Select all categorical variables again
df_categorical = df.select_dtypes(include=['object'])

# Initialize Label Encoder
le = preprocessing.LabelEncoder()

# Apply label encoder to the categorical DataFrame
df_categorical = df_categorical.apply(le.fit_transform)
df_categorical.head()

# Drop the original categorical columns from the DataFrame
df = df.drop(df_categorical.columns, axis=1)

# Concatenate the DataFrame with the encoded categorical columns
df = pd.concat([df, df_categorical], axis=1)
df.head()

# Convert the target variable 'income' to categorical type
df['income'] = df['income'].astype('category')

# Importing train_test_split to split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Define features (independent variables) and target (dependent variable)
X = df.drop('income', axis=1)  # Features
y = df['income']  # Target variable

# Display the first 3 rows of the features
X.head(3)

# Split the dataset into training and testing sets with 30% of the data as the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=99)
X_train.head()

# Importing the Decision Tree Classifier from sklearn
from sklearn.tree import DecisionTreeClassifier

# Initialize and fit the decision tree classifier with a maximum depth of 5
dt_default = DecisionTreeClassifier(max_depth=5)
dt_default.fit(X_train, y_train)

# Import classification report and confusion matrix from sklearn metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Make predictions using the test set
y_pred_default = dt_default.predict(X_test)

# Print the classification report showing precision, recall, and f1-score
print(classification_report(y_test, y_pred_default))

# Print the confusion matrix
print(confusion_matrix(y_test, y_pred_default))

# Print the accuracy score of the model
print(accuracy_score(y_test, y_pred_default))

# Conclusion: The confusion matrix indicates the model's performance on different classes.
# It may require techniques such as resampling or threshold adjustment to improve performance on the minority class.
