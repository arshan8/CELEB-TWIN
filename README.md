<pre>


    #1
import csv

def find_s(data):
    # Initialize hypothesis with '?' for all attributes except target
    hypothesis = ['?'] * (len(data[0]) - 1)
    
    for example in data[1:]:  # skip header row
        if example[-1] == 'Yes':  # Only consider positive examples
            for i in range(len(hypothesis)):
                if hypothesis[i] == '?':
                    hypothesis[i] = example[i]
                elif hypothesis[i] != example[i]:
                    hypothesis[i] = '?'
    return hypothesis

# Read CSV and run
with open(r"C:\Users\ARSHAN\Downloads\Weather.csv", 'r') as file:
    data = list(csv.reader(file))
    result = find_s(data)
    print("Final Hypothesis:", result)





#2
import csv

def candidate_elimination(data):
    # Initialize specific and general hypotheses
    specific_h = ['?' for _ in range(len(data[0])-1)]
    general_h = [['?' for _ in range(len(data[0])-1)] for _ in range(len(data[0])-1)]
    
    for example in data:
        if example[-1] == 'Yes':  # Positive example
            for i in range(len(specific_h)):
                if specific_h[i] == '?' or specific_h[i] == example[i]:
                    specific_h[i] = example[i]
                else:
                    specific_h[i] = '?'
        else:  # Negative example
            for i in range(len(specific_h)):
                if specific_h[i] != example[i]:
                    general_h[i][i] = specific_h[i] if specific_h[i] != '?' else '?'
                else:
                    general_h[i][i] = '?'
    
    return specific_h, [g for g in general_h if g != ['?' for _ in range(len(data[0])-1)]]

# Read CSV and run
with open(r"C:\Users\ARSHAN\Downloads\Weather.csv", 'r') as file:
    data = list(csv.reader(file))
    specific, general = candidate_elimination(data)
    print("Specific Hypothesis:", specific)
    print("General Hypothesis:", general)





#3

    import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# ---- Read CSV ----
df = pd.read_csv(r"C:\Users\ARSHAN\Downloads\playtennis.csv")

# Split into features and target
X = df.iloc[:, :-1]   # all columns except last
y = df.iloc[:, -1]    # last column (target)

# ---- Train Decision Tree ----
tree = DecisionTreeClassifier(criterion='entropy')
tree.fit(X, y)

# ---- Accuracy ----
print("Training Accuracy:", tree.score(X, y))

# ---- Plot the tree ----
plt.figure(figsize=(10, 6))
plot_tree(tree, feature_names=df.columns[:-1], class_names=['No', 'Yes'], filled=True)
plt.show()





#4

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import numpy as np

# XOR dataset
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
y = np.array([0, 1, 1, 0])   # labels

# ANN: 1 hidden layer (2 neurons), ReLU, sigmoid output
mlp = MLPClassifier(hidden_layer_sizes=(2,),
                    activation='relu',
                    solver='sgd',
                    learning_rate_init=0.1,
                    max_iter=1000,
                    random_state=0, verbose = 1)

# Train
mlp.fit(X, y)

# Predictions
y_pred = mlp.predict(X)

# Classification report
print("Classification Report:\n")
print(classification_report(y, y_pred))




#5

    import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
data = pd.read_csv(r"C:\Users\ARSHAN\Downloads\DBetes.csv")  # update path if needed

# Features (X) and target (y)
X = data.iloc[:, :-1]   # all columns except last
y = data.iloc[:, -1]    # last column = Outcome

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=0
)

# Train Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Precision, Recall, F1
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))





#6
    import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Load dataset from CSV
data = pd.read_csv("6a.csv")
heart_disease = data  # Use the loaded data directly

# Define the Bayesian model structure
model = BayesianModel([
    ('age', 'Lifestyle'),
    ('Gender', 'Lifestyle'),
    ('Family', 'heartdisease'),
    ('diet', 'cholestrol'),
    ('Lifestyle', 'diet'),
    ('cholestrol', 'heartdisease')
])  # Removed duplicate ('diet', 'cholestrol')

# Fit the model with the data
model.fit(heart_disease, estimator=MaximumLikelihoodEstimator)

# Create inference object
HeartDisease_infer = VariableElimination(model)

# User input prompts
print('For Age enter: SuperSeniorCitizen:0, SeniorCitizen:1, MiddleAged:2, Youth:3, Teen:4')
print('For Gender enter: Male:0, Female:1')
print('For Family History enter: Yes:1, No:0')
print('For Diet enter: High:0, Medium:1')
print('For Lifestyle enter: Athlete:0, Active:1, Moderate:2, Sedentary:3')
print('For Cholesterol enter: High:0, BorderLine:1, Normal:2')

# Get evidence from user
evidence = {
    'age': int(input('Enter Age (0-4): ')),
    'Gender': int(input('Enter Gender (0-1): ')),
    'Family': int(input('Enter Family History (0-1): ')),
    'diet': int(input('Enter Diet (0-1): ')),
    'Lifestyle': int(input('Enter Lifestyle (0-3): ')),
    'cholestrol': int(input('Enter Cholesterol (0-2): '))
}

# Query the probability of heart disease
q = HeartDisease_infer.query(variables=['heartdisease'], evidence=evidence)

# Print results
print("\nProbability of having heart disease:")
print(f"P(Heart Disease = Yes) = {q.values[1]}")  # Yes is index 1
print(f"P(Heart Disease = No) = {q.values[0]}")   # No is index 0








#7
    import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Set random seed for reproducibility
np.random.seed(110)

# Generate synthetic data
red_mean, red_std = 3, 0.8    # Red cluster mean and standard deviation
blue_mean, blue_std = 7, 1    # Blue cluster mean and standard deviation
red = np.random.normal(red_mean, red_std, size=40)   # Red data points
blue = np.random.normal(blue_mean, blue_std, size=40) # Blue data points
both_colours = np.sort(np.concatenate((red, blue)))   # Combine and sort
y = np.zeros(len(both_colours))                      # Y-axis placeholder

# Perform K-Means with 2 clusters
kmeans = KMeans(n_clusters=2)
kmeans.fit(both_colours.reshape(-1, 1))  # Reshape for 1D data
labels = kmeans.labels_  # Cluster assignments

# Elbow curve to find optimal number of clusters
Nc = range(1, 5)  # Test 1 to 4 clusters
scores = [KMeans(n_clusters=i).fit(both_colours.reshape(-1, 1)).score(both_colours.reshape(-1, 1)) for i in Nc]

# Plot Elbow curve
plt.figure(figsize=(8, 5))
plt.plot(Nc, scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()

# Plot clustering result
plt.figure(figsize=(8, 5))
plt.scatter(both_colours, y, c=labels, cmap='bwr')
plt.xlabel('Data Points')
plt.ylabel('None')
plt.title('2 Cluster K-Means')
plt.show()




#8
    
from sklearn.datasets import load_iris
iris = load_iris()
print("Feature Names:",iris.feature_names)
print("Iris Data:",iris.data)
print("Target Names:",iris.target_names)
print("Target:",iris.target)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
      iris.data, iris.target, test_size = .25)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)

print(" Accuracy=",clf.score(X_test, y_test))

print("Predicted Data")
print(clf.predict(X_test))
prediction=clf.predict(X_test)
print("Test data :")
print(y_test)

diff=prediction-y_test
print("Result is ")
print(diff)
print('Total no of samples misclassied =', sum(abs(diff))) 








#9

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load data
data = pd.read_csv(r"C:\Users\ARSHAN\Downloads\LR.csv")
X = data['colA'].values
y = data['colB'].values

# Simple LWR: Use a basic weighted average of nearby points
def simple_lwr(x_query, X, y, window_size=1.0):
    # Find weights based on distance (simpler weighting)
    distances = np.abs(X - x_query)
    weights = np.exp(-distances / window_size)  # Simple exponential decay
    
    # Weighted average for prediction
    weighted_y = y * weights
    total_weight = np.sum(weights)
    return np.sum(weighted_y) / total_weight if total_weight > 0 else np.mean(y)

# Generate prediction points
X_pred = np.linspace(min(X), max(X), 100)
y_pred = np.array([simple_lwr(x, X, y, window_size=1.0) for x in X_pred])

# Plot
plt.scatter(X, y, color='green', label='Original Data')
plt.plot(X_pred, y_pred, color='red', label='LWR Fit')
plt.xlabel('ColA')
plt.ylabel('ColB')
plt.title('Simple Locally Weighted Regression')
plt.legend()
plt.show()import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load data
data = pd.read_csv(r"C:\Users\ARSHAN\Downloads\LR.csv")
X = data['colA'].values
y = data['colB'].values

# Simple LWR: Use a basic weighted average of nearby points
def simple_lwr(x_query, X, y, window_size=1.0):
    # Find weights based on distance (simpler weighting)
    distances = np.abs(X - x_query)
    weights = np.exp(-distances / window_size)  # Simple exponential decay
    
    # Weighted average for prediction
    weighted_y = y * weights
    total_weight = np.sum(weights)
    return np.sum(weighted_y) / total_weight if total_weight > 0 else np.mean(y)

# Generate prediction points
X_pred = np.linspace(min(X), max(X), 100)
y_pred = np.array([simple_lwr(x, X, y, window_size=1.0) for x in X_pred])

# Plot
plt.scatter(X, y, color='green', label='Original Data')
plt.plot(X_pred, y_pred, color='red', label='LWR Fit')
plt.xlabel('ColA')
plt.ylabel('ColB')
plt.title('Simple Locally Weighted Regression')
plt.legend()
plt.show()






