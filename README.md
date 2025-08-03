
# 1

import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=10):
        self.weights = np.zeros(input_size + 1)  # +1 for bias
        self.lr = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0
    
    

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activation(summation)

    def train(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                pred = self.predict(xi)
                error = target - pred
                self.weights[1:] += self.lr * error * xi
                self.weights[0] += self.lr * error
#wj = wj + learning_rate * (true_output - predicted_output) * xj
#bias = bias + learning_rate * (true_output - predicted_output)


    def test(self, X):
        return [self.predict(xi) for xi in X]

    
    
    
def or_gate():
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([0, 1, 1, 1])

    p = Perceptron(input_size=2)
    p.train(X, y)
    
    print("OR Gate Output:")
    for xi in X:
        print(f"{xi} -> {p.predict(xi)}")
        
    return p,X,y



def and_gate():
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([0, 0, 0, 1])

    p = Perceptron(input_size=2)
    p.train(X, y)
    
    print("AND Gate Output:")
    for xi in X:
        print(f"{xi} -> {p.predict(xi)}")
        
    return p,X,y






import matplotlib.pyplot as plt
import numpy as np
def plot_decision_boundary(perceptron, X, y):
    # Generate grid points
    x1 = np.linspace(-1, 2, 100)
    x2 = np.linspace(-1, 2, 100)
    xx, yy = np.meshgrid(x1, x2)

    # Predict on grid
    Z = np.array([perceptron.predict([a, b]) for a, b in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)

    # Plot
    plt.contourf(xx, yy, Z, alpha=0.5)
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.show()

p,X,y = and_gate()
plot_decision_boundary(p, X, y)








from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# Load data
wine = load_wine()
X, y = wine.data, wine.target

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dimensionality reduction for visualization
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Train MLP
clf = MLPClassifier(hidden_layer_sizes=(8,), activation='relu', max_iter=10000)
clf.fit(X_train, y_train)

# Accuracy
print(f"Training Accuracy: {clf.score(X_train, y_train):.2f}")
print(f"Test Accuracy: {clf.score(X_test, y_test):.2f}")

# Plot decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title("MLP Wine Classification (PCA 2D)")
    plt.show()

plot_decision_boundary(clf, X_reduced, y)

















import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# Generate simple raw random data for two crab species
# Class 0: Crab A
X0 = np.random.rand(100, 2) + 1  # values roughly between 1 and 2
y0 = np.zeros(100)

# Class 1: Crab B
X1 = np.random.rand(100, 2) + 3  # values roughly between 3 and 4
y1 = np.ones(100)

# Combine data
X = np.vstack((X0, X1))
y = np.hstack((y0, y1))

# Train MLP classifier
clf = MLPClassifier(hidden_layer_sizes=(8,), activation='relu', max_iter=10000)
clf.fit(X, y)

# Print predictions for first 5 samples
print("First 5 predictions:")
for i in range(5):
    print(f"{X[i]} -> {clf.predict([X[i]])[0]}")

# Plot decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title("MLP Crab Classification Decision Boundary")
    plt.show()

plot_decision_boundary(clf, X, y)


















from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt

# XOR data
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

# Train MLP
clf = MLPClassifier(hidden_layer_sizes=(8,), activation='relu', max_iter=10000)
clf.fit(X, y)

# Predict
print("XOR Output:")
for xi in X:
    print(f"{xi} -> {clf.predict([xi])[0]}")

# Plot decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title("MLP XOR Classification Decision Boundary")
    plt.show()

plot_decision_boundary(clf, X, y)










import numpy as np

# Define your scalar function f(x, y)
def f(x, y):
    return x**2 * y + y**3

# Compute gradient (Jacobian: 1x2)
def compute_jacobian(f, x, y, h=1e-5):
    df_dx = (f(x + h, y) - f(x - h, y)) / (2*h)
    df_dy = (f(x, y + h) - f(x, y - h)) / (2*h)
    return np.array([[df_dx, df_dy]])  # shape (1, 2)

# Compute Hessian (2x2)
def compute_hessian(f, x, y, h=1e-5):
    d2f_dx2 = (f(x + h, y) - 2*f(x, y) + f(x - h, y)) / (h**2)
    d2f_dy2 = (f(x, y + h) - 2*f(x, y) + f(x, y - h)) / (h**2)
    d2f_dxdy = (
        f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)
    ) / (4 * h**2)

    return np.array([
        [d2f_dx2, d2f_dxdy],
        [d2f_dxdy, d2f_dy2]
    ])

# Example usage
x_val, y_val = 1.0, 2.0

jacobian = compute_jacobian(f, x_val, y_val)
hessian = compute_hessian(f, x_val, y_val)

print("Jacobian (1x2):", jacobian)
print("Hessian (2x2):")
print(hessian)











