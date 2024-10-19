# Import necessary libraries
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
from sklearn.model_selection import train_test_split  # For splitting dataset into train and test sets
from sklearn.preprocessing import StandardScaler, LabelEncoder  # For scaling features and encoding categorical variables
from sklearn.decomposition import PCA  # For Principal Component Analysis
from sklearn.linear_model import LogisticRegression  # For logistic regression model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # For model evaluation metrics
from sklearn.impute import SimpleImputer  # For handling missing values

# Load the Adult Census Income dataset
df = pd.read_csv('adult.csv', header=None)  # Read the CSV file into a DataFrame without headers

# Assign column names to the DataFrame for clarity
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education.num',
           'marital.status', 'occupation', 'relationship', 'race',
           'sex', 'capital.gain', 'capital.loss', 'hours.per.week',
           'native.country', 'income']  # List of column names for better understanding
df.columns = columns  # Assign the column names to the DataFrame

# Handle missing values
df = df.replace(' ?', np.nan)  # Replace any occurrence of '?' in the dataset with NaN (Not a Number)
df.dropna(inplace=True)  # Drop rows that contain any NaN values, cleaning the dataset

# Encode categorical features using Label Encoding
label_encoders = {}  # Initialize a dictionary to store label encoders for each categorical column
for column in ['workclass', 'education', 'marital.status',
               'occupation', 'relationship', 'race', 'sex', 
               'native.country', 'income']:  # List of categorical columns to encode
    le = LabelEncoder()  # Create a LabelEncoder object
    df[column] = le.fit_transform(df[column])  # Fit and transform the categorical data to numerical values
    label_encoders[column] = le  # Store the label encoder for potential inverse transformation later

# Separate features (X) and target variable (y)
X = df.drop('income', axis=1)  # Features: all columns except 'income' (input variables)
y = df['income']  # Target variable: 'income' column (output variable)

# Check if all columns in X are numeric
print("Data Types in X before Imputation:")  # Print message indicating the next output
print(X.dtypes)  # Display the data types of the columns in X

# Ensure all columns in X are numeric
X = X.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric, coercing errors to NaN

# Check for any non-numeric values after conversion
print("\nData Types in X after ensuring numeric:")  # Print message indicating the next output
print(X.dtypes)  # Display the data types of the columns in X after conversion

# Impute missing values in features (if any)
imputer = SimpleImputer(strategy='mean')  # Initialize imputer with mean strategy for filling NaN values
X_imputed = imputer.fit_transform(X)  # Fit the imputer and transform the data, filling NaN values

# Scale the features to have mean = 0 and variance = 1
scaler = StandardScaler()  # Initialize the scaler
X_scaled = scaler.fit_transform(X_imputed)  # Scale the imputed data to standardize features

# Apply PCA for dimensionality reduction
pca = PCA(n_components=5)  # Initialize PCA to reduce to 5 principal components
X_pca = pca.fit_transform(X_scaled)  # Fit PCA on the scaled data and transform it to the new feature space

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)  # Split data with 30% for testing

# Train a Logistic Regression model on the reduced dataset
model = LogisticRegression()  # Initialize the logistic regression model
model.fit(X_train, y_train)  # Fit the model on the training data

# Make predictions on the test set
y_pred = model.predict(X_test)  # Generate predictions for the test set

# Evaluate the model's performance
print("Accuracy Score:", accuracy_score(y_test, y_pred))  # Print the accuracy score of the model
print("Confusion Matrix:")  
print(confusion_matrix(y_test, y_pred))  # Print the confusion matrix to see classification results
print("Classification Report:")  
print(classification_report(y_test, y_pred))  # Print a detailed classification report with precision, recall, and F1 scores

# Additional analysis: Explained variance ratio of PCA components
explained_variance = pca.explained_variance_ratio_  # Retrieve the explained variance ratio for each principal component
print("\nExplained Variance Ratio of PCA components:")  # Print message indicating the next output
for i, var in enumerate(explained_variance):  # Loop through each component's variance ratio
    print(f"Principal Component {i + 1}: {var:.2f}")  # Print the explained variance ratio for each component
