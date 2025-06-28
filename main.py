"""
Linear regression is a statistical method used to model the relationship between a dependent variable and one or more
independent variables by fitting a linear equation to the observed data.(In simple terms, it tries to find the "best fit"
line or plane that represents the relationship between the varaiables)  
"""

from sklearn.linear_model import LinearRegression
import numpy as np

#Sample data: X = feature, y = target
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

#Create and training the model
model = LinearRegression()
model.fit(X, y)

#Predict
prediction = model.predict([[6]])
print(f"Prediction for 6: {prediction[0]}")


"""
Classification with logistic regresssion 
Supervised machine learnikng algorithm primarily used for calssification tasks, particularly
binary classification where the outcome has two possible classes .It moldels the probability of the dependent
variable belonging to a specific class using a logistic function.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

#Load the Iris dataset
data = load_iris()
X, y = data.data, data.target

#Train a logistic regression model (binary classsification: class 0 vs class 1)
model = LogisticRegression(max_iter=200)
model.fit(X[:100], y[:100])

#Predict
print(f"Predicted class: {model.predict([[5.1, 3.5, 1.4, 0.2]])[0]}")

"""
K-Means Clustering(unsupervised Learning)
K-means clustering is an unsupervised  machine learning algorithm used to grolup data points into K clustets, 
where each data point  belongs to the cluster with the neares mean(centroid).It`s unsupervised because it doenst
require labeled data; the algorithm discovers  patterns and structures within the data itself. THe goal is to minimize
the variance within each cluster, resulting in compact and well-separated clusters
"""

from sklearn.cluster import KMeans
import numpy as np 

#Sample data 
X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

#Cluster into 2 groups
kmeans = KMeans(n_clusters=2).fit(X)

# Predict clusters
print(f"Cluster labels: {kmeans.labels_}")

"""
Decision Tree Classifier
A Machine Learning algorithm that builds a tree-like structure to predict the class label of a data instance.It
works by asking a series of questions (tests) based on the features of the data, with each answer leading to a 
different branch of the tree. Ultimately, the path trough the tree leads to a leaf node, wich represents the predi-
cted class. 
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

#Load data 
data = load_iris()
X, y = data.data, data.target

#Traind a decision tree
model = DecisionTreeClassifier()
model.fit(X, y)

#Predict
print(f"Predicted class: {model.predict([[5.0, 3.0, 1.0, 0.10]])[0]}")

"""
Neural Network with TensorFlow/Keras
You can build an artificial nural network using its high-level API, Keras. Keras simplifies the proxess of designing and training
ANNs by providing pre-built components like layers, loss functions, and optimizers 
"""

import tensorFlow as tf
from tensorflow.keras.layers import Dense

#Simple sequential model 
model = tf.keras.Sequential([
    Dense(10, activation='relu', input_shape=(4,)),
    Dense(3, activation='softmax') # classes for iris 
])

#Compile and tain
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X, y, epochs=10)

#Predict
print(f"Predicted probabilities: {model.predict([[5.1, 3.5, 1.4, 0.2]])}")