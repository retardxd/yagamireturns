# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
from sklearn.model_selection import train_test_split  # To split the dataset into training and testing sets
from sklearn.ensemble import AdaBoostClassifier  # The AdaBoost algorithm for classification
from sklearn.tree import DecisionTreeClassifier  # Decision tree as the base estimator
from sklearn.metrics import classification_report, accuracy_score  # Metrics for model evaluation

# Load the Adult Census Income dataset
df = pd.read_csv('adult.csv')  # Read the CSV file into a DataFrame
print(df.head())  # Display the first five rows of the DataFrame to understand its structure

# Rename columns for clarity
df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
               'marital-status', 'occupation', 'relationship', 'race', 
               'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 
               'native-country', 'income']  # Assign meaningful names to the columns

# Preprocess the dataset
df.replace(' ?', pd.NA, inplace=True)  # Replace missing values represented as '?' with NaN
df.dropna(inplace=True)  # Remove rows with any NaN values

# Strip whitespace from string columns
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Define categorical columns for encoding
categorical_columns = ['workclass', 'education', 'marital-status', 
                       'occupation', 'relationship', 'race', 'sex', 
                       'native-country', 'income']

# Initialize a dictionary to store label encoders for each categorical column
label_encoders = {}
for col in categorical_columns:  # Loop through each categorical column
    le = LabelEncoder()  # Create a LabelEncoder object
    df[col] = le.fit_transform(df[col])  # Fit and transform the categorical data to numerical values
    label_encoders[col] = le  # Store the label encoder for potential inverse transformation later

# Define features (X) and target variable (y)
X = df.drop('income', axis=1)  # Features: all columns except 'income'
y = df['income']  # Target variable: 'income' column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the AdaBoost classifier with a decision tree as the base estimator
base_estimator = DecisionTreeClassifier(max_depth=1)  # Decision stump
boosting_model = AdaBoostClassifier(estimator=base_estimator, n_estimators=100, random_state=42)

# Fit the AdaBoost model to the training data
boosting_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = boosting_model.predict(X_test)

# Print evaluation metrics
print(f"Final Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")  # Calculate and print the accuracy score
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")  # Print classification metrics
