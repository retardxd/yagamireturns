# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
from sklearn.model_selection import train_test_split  # To split the dataset into training and testing sets
from sklearn.ensemble import RandomForestClassifier  # The Random Forest algorithm for classification
from sklearn.preprocessing import LabelEncoder  # To encode categorical variables into numerical format
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

# Function to evaluate model performance using bootstrapping
def bootstrap_evaluate(X, y, n_bootstraps=100):
    accuracies = []  # List to store accuracy for each bootstrap iteration
    for i in range(n_bootstraps):  # Loop for the number of bootstraps
        indices = np.random.randint(0, len(X), len(X))  # Randomly sample indices with replacement
        bootstrap_X, bootstrap_y = X.iloc[indices], y.iloc[indices]  # Create bootstrap samples
        # Split the bootstrap samples into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(bootstrap_X, 
                                                            bootstrap_y, 
                                                            test_size=0.2, 
                                                            random_state=42)
        # Initialize the Random Forest Classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)  
        rf.fit(X_train, y_train)  # Fit the model on the training data
        y_pred = rf.predict(X_test)  # Predict on the test set
        accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
        accuracies.append(accuracy)  # Store accuracy in the list
    mean_accuracy = np.mean(accuracies)  # Calculate mean accuracy across bootstraps
    std_accuracy = np.std(accuracies)  # Calculate standard deviation of accuracies
    return mean_accuracy, std_accuracy  # Return mean and std deviation of accuracy

# Evaluate the model using bootstrapping
mean_acc, std_acc = bootstrap_evaluate(X, y, n_bootstraps=30)

# Split the entire dataset into training and testing sets for final evaluation
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X, y, 
                                                                           test_size=0.2, 
                                                                           random_state=42)
# Initialize the Random Forest Classifier for final model training
rf_final = RandomForestClassifier(n_estimators=100, random_state=42)  
rf_final.fit(X_train_final, y_train_final)  # Fit the model on the training data
y_pred_final = rf_final.predict(X_test_final)  # Predict on the test set

# Print final model evaluation metrics
print(f"Final Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test_final, y_pred_final)}")  # Calculate and print the accuracy score
print(f"Classification Report:\n{classification_report(y_test_final, y_pred_final)}")  # Print classification metrics
print(f"Mean Accuracy (Bootstrapping): {mean_acc}")  # Print mean accuracy from bootstrapping
print(f"Standard Deviation of Accuracy (Bootstrapping): {std_acc}")  # Print standard deviation of accuracy
