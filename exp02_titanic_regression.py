# Import necessary libraries

#EXP02

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, RocCurveDisplay

# Load the Titanic dataset
df = pd.read_csv('titanic_data.csv')

# Display the first five rows of the dataset
df.head()

# Display information about the dataset, including data types and missing values
df.info()

# Select relevant features for analysis
df = df[['Survived', 'Age', 'Sex', 'Pclass']]

# Apply one-hot encoding to categorical variables (Sex and Pclass)
df = pd.get_dummies(df, columns=['Sex', 'Pclass'])

# Drop rows with missing values
df.dropna(inplace=True)

# Define features (X) and target variable (y)
x = df.drop('Survived', axis=1)  # Features
y = df['Survived']                # Target variable

# Split the dataset into training and test sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)

# Initialize the Logistic Regression model
model = LogisticRegression(random_state=0)

# Fit the model on the training data
model.fit(x_train, y_train)

# Evaluate the model's accuracy on the test set
accuracy = model.score(x_test, y_test)
print(f'Accuracy on test set: {accuracy:.2%}')  # Print accuracy as a percentage

# Perform 5-fold cross-validation to assess model stability
cv_score = cross_val_score(model, x, y, cv=5).mean()
print(f'Cross-Validation Score: {cv_score:.2%}')  # Print mean cross-validation score

# Make predictions on the test set
y_predicted = model.predict(x_test)

# Calculate and display the confusion matrix
cm = confusion_matrix(y_test, y_predicted)
print('Confusion Matrix:\n', cm)

# Display the confusion matrix visually
ConfusionMatrixDisplay.from_estimator(model, x_test, y_test)

# Generate and print a classification report
report = classification_report(y_test, y_predicted)
print('Classification Report:\n', report)

# Visualize the ROC curve
RocCurveDisplay.from_estimator(model, x_test, y_test)

# Example prediction: Predict survival for a female passenger aged 30 in 1st class
female = [[30, 1, 0, 1, 0, 0]]  # Features: Age, Sex_female, Sex_male, Pclass_1, Pclass_2, Pclass_3
predicted_class = model.predict(female)[0]
print(f'Predicted class for female passenger: {predicted_class}')  # Print predicted class (0 or 1)

# Calculate the probability of survival for the same female passenger
probability = model.predict_proba(female)[0][1]
print(f'Probability of survival: {probability:.1%}')  # Print probability of survival as a percentage
