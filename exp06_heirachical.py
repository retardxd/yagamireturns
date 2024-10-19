# Import necessary libraries
import os  # For interacting with the operating system
import pandas as pd  # For data manipulation and analysis
import matplotlib.pyplot as plt  # For data visualization
from sklearn.preprocessing import normalize  # For normalizing data
import scipy.cluster.hierarchy as shc  # For hierarchical clustering
from sklearn.cluster import AgglomerativeClustering  # For agglomerative clustering

# Load the Wholesale Customers dataset
data = pd.read_csv('Wholesale customers data.csv')  # Read the CSV file into a DataFrame
print(data.head())  # Display the first five rows of the DataFrame to understand its structure

# Normalize the dataset for effective clustering
data_scaled = normalize(data)  # Normalize the data to bring all features to the same scale
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)  # Convert the normalized data back to a DataFrame with original columns
print(data_scaled.head())  # Display the first five rows of the scaled data

# Create a dendrogram to visualize the hierarchical clustering
plt.figure(figsize=(10, 7))  # Set the figure size for the plot
plt.title("Dendrograms")  # Set the title for the plot
d = shc.dendrogram(shc.linkage(data_scaled, method='ward'))  # Generate the dendrogram using Ward's method

# Add a horizontal line to indicate the cutoff for clusters
plt.axhline(y=6, color='r', linestyle='--')  # Draw a dashed red line at y=6 to show the cutoff for clusters

# Apply Agglomerative Clustering
cluster = AgglomerativeClustering(n_clusters=2, linkage='ward')  # Create an AgglomerativeClustering model with 2 clusters
print(cluster.fit_predict(data_scaled))  # Fit the model on the scaled data and predict cluster labels

# Visualize the clusters in a scatter plot
plt.figure(figsize=(10, 7))  # Set the figure size for the scatter plot
plt.scatter(data_scaled['Milk'], data_scaled['Grocery'], c=cluster.labels_)  # Plot 'Milk' vs 'Grocery', colored by cluster labels
plt.title("Clusters based on Milk and Grocery Spending")  # Set the title for the scatter plot
plt.xlabel("Milk Spending (normalized)")  # Set the x-axis label
plt.ylabel("Grocery Spending (normalized)")  # Set the y-axis label
plt.show()  # Display the plot
    