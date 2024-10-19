# Importing necessary libraries
import pandas as pd  # for data manipulation
import numpy as np  # for numerical operations
import matplotlib.pyplot as plt  # for plotting graphs
import seaborn as sns  # for advanced visualizations
from sklearn.model_selection import train_test_split  # for splitting dataset
from sklearn.linear_model import LinearRegression  # for applying linear regression
from sklearn.metrics import mean_squared_error, r2_score  # for evaluating the model
from scipy import stats  # for statistical operations (if required)

# Load the dataset into a pandas DataFrame
Boston = pd.read_csv('Boston.csv')

# Display the first few rows of the dataset to understand its structure
Boston.head()

# Get a summary of the dataset, including data types and non-null values
Boston.info()

# Statistical summary of the dataset (mean, standard deviation, etc.)
Boston.describe()

# Check for missing values in the dataset
missing_values = Boston.isna().sum()
print(missing_values)

# List of columns where missing values are found
na_columns = ['crim', 'zn', 'indus', 'chas', 'age', 'lstat']

# Fill missing values in the specified columns with the mean of the respective column
Boston[na_columns] = Boston[na_columns].fillna(Boston.mean())

# Print the updated dataset to verify missing values are handled
print(Boston)

# Separate the target variable (medv) from the dataset
target = Boston['medv']
print(target)

# Define the feature set X by dropping the target column (medv)
X = Boston.drop(["medv"], axis=1)

# Define the target variable y as 'medv'
y = Boston["medv"]

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a Linear Regression model instance
model = LinearRegression()

# Fit the model using the training data
model.fit(X_train, y_train)

# Make predictions using the test data
y_pred = model.predict(X_test)

# Plot a scatter plot to compare the actual values (y_test) with the predicted values (y_pred)
plt.scatter(y_test, y_pred, c='red')
plt.xlabel("medv")  # Label for the x-axis (True values)
plt.ylabel("Predicted value")  # Label for the y-axis (Predicted values)
plt.title("True value vs predicted value : Linear Regression")  # Title of the plot
plt.show()  # Display the plot

# Set the figure size for Seaborn plots
sns.set(rc={'figure.figsize':(11.7,8.27)})

# Plot the distribution of the target variable (medv) to check its spread and skewness
sns.distplot(Boston['medv'], bins=30)
plt.show()

# Generate the correlation matrix to check the relationships between features
correlation_matrix = Boston.corr().round(2)

# Visualize the correlation matrix using a heatmap (annot=True to display correlation values)
sns.heatmap(data=correlation_matrix, annot=True)

# Plot relationships between selected features ('lstat' and 'rm') and the target variable (medv)
plt.figure(figsize=(20, 5))  # Set the figure size for the plot
features = ['lstat', 'rm']  # List of features to plot
target = Boston['medv']  # The target variable

# Loop through the selected features and create scatter plots
for i, col in enumerate(features):
    plt.subplot(1, len(features), i+1)  # Create subplots for each feature
    x = Boston[col]  # Feature values
    y = target  # Target values (medv)
    plt.scatter(x, y, marker='o')  # Scatter plot
    plt.title(col)  # Title of each subplot (feature name)
    plt.xlabel(col)  # Label for x-axis
    plt.ylabel('medv')  # Label for y-axis

# Create a new feature set X using only 'lstat' and 'rm' columns
X = pd.DataFrame(np.c_[Boston['lstat'], Boston['rm']], columns=['lstat','rm'])

# The target variable remains 'medv'
Y = Boston['medv']

# Split the data again into training (80%) and testing (20%) sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

# Print the shapes of training and testing sets to ensure correct splitting
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# Create another instance of Linear Regression for the smaller feature set
lin_model = LinearRegression()

# Fit the model using the training data
lin_model.fit(X_train, Y_train)

# Make predictions using the training set
y_train_predict = lin_model.predict(X_train)

# Calculate the Root Mean Squared Error (RMSE) for the training set
rmse_train = np.sqrt(mean_squared_error(Y_train, y_train_predict))

# Calculate the R² score for the training set
r2_train = r2_score(Y_train, y_train_predict)

# Make predictions using the test set
y_test_predict = lin_model.predict(X_test)

# Calculate the RMSE for the test set
rmse_test = np.sqrt(mean_squared_error(Y_test, y_test_predict))

# Calculate the R² score for the test set
r2_test = r2_score(Y_test, y_test_predict)

# Print the model performance metrics for the training set
print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse_train))  # Display RMSE for the training set
print('R2 score is {}'.format(r2_train))  # Display R² score for the training set

# Print the model performance metrics for the testing set
print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse_test))  # Display RMSE for the test set
print('R2 score is {}'.format(r2_test))  # Display R² score for the test set
