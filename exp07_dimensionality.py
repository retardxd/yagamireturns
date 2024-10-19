import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer

# Load the Adult Census Income dataset
df = pd.read_csv('adult.csv', header=None)

# Assign column names to the DataFrame for clarity
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education.num',
           'marital.status', 'occupation', 'relationship', 'race',
           'sex', 'capital.gain', 'capital.loss', 'hours.per.week',
           'native.country', 'income']
df.columns = columns

# Handle missing values
df = df.replace(' ?', np.nan)  # Replace '?' with NaN
df.dropna(inplace=True)  # Drop rows with any NaN values

# Encode categorical features using Label Encoding
label_encoders = {}  # Initialize a dictionary to store label encoders for each categorical column
for column in ['workclass', 'education', 'marital.status',
               'occupation', 'relationship', 'race', 'sex', 
               'native.country', 'income']:
    le = LabelEncoder()  # Create a LabelEncoder object
    df[column] = le.fit_transform(df[column])  # Fit and transform the categorical data to numerical values
    label_encoders[column] = le  # Store the label encoder for potential inverse transformation later

# Separate features (X) and target variable (y)
X = df.drop('income', axis=1)  # Features: all columns except 'income'
y = df['income']  # Target variable: 'income' column

# Check if all columns in X are numeric
print("Data Types in X before Imputation:")
print(X.dtypes)

# Ensure all columns in X are numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Check for any non-numeric values after conversion
print("\nData Types in X after ensuring numeric:")
print(X.dtypes)

# Impute missing values in features (if any)
imputer = SimpleImputer(strategy='mean')  # Initialize imputer with mean strategy
X_imputed = imputer.fit_transform(X)  # Fit the imputer and transform the data

# Scale the features to have mean = 0 and variance = 1
scaler = StandardScaler()  # Initialize the scaler
X_scaled = scaler.fit_transform(X_imputed)  # Scale the imputed data

# Apply PCA for dimensionality reduction
pca = PCA(n_components=5)  # Initialize PCA to reduce to 5 principal components
X_pca = pca.fit_transform(X_scaled)  # Fit and transform the scaled data

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Train a Logistic Regression model on the reduced dataset
model = LogisticRegression()  # Initialize the logistic regression model
model.fit(X_train, y_train)  # Fit the model on the training data

# Make predictions on the test set
y_pred = model.predict(X_test)  # Generate predictions

# Evaluate the model's performance
print("Accuracy Score:", accuracy_score(y_test, y_pred))  # Print the accuracy score of the model
print("Confusion Matrix:")  
print(confusion_matrix(y_test, y_pred))  # Print the confusion matrix to see the classification results
print("Classification Report:")  
print(classification_report(y_test, y_pred))  # Print a detailed classification report with precision, recall, and F1 scores

# Additional analysis: Explained variance ratio of PCA components
explained_variance = pca.explained_variance_ratio_
print("\nExplained Variance Ratio of PCA components:")
for i, var in enumerate(explained_variance):
    print(f"Principal Component {i + 1}: {var:.2f}")
