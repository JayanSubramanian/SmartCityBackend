import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Step 1: Generate Synthetic Data
def generate_synthetic_data():
    np.random.seed(42)
    
    # Features: AE counts, AE energy, signal strength
    # Healthy (class 0): Lower AE counts, energy, and signal strength
    # Leakage (class 1): Higher AE counts, energy, and signal strength
    
    n_samples = 1000  # Number of samples
    n_features = 3    # AE counts, AE energy, signal strength

    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_features, 
                               n_redundant=0, n_classes=2, flip_y=0.05, class_sep=1.5, random_state=42)
    return X, y

# Step 2: Preprocess the Data
X, y = generate_synthetic_data()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Step 3: Train SVM
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)

# Step 4: Evaluate the Model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print Evaluation Metrics
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)

# Step 5: Visualize Decision Boundaries (for the first two features)
def plot_decision_boundaries(X, y, model, title):
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict the class for each point in the mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.xlabel('Feature 1 (Normalized)')
    plt.ylabel('Feature 2 (Normalized)')
    plt.show()

plot_decision_boundaries(X_scaled[:, :2], y, svm_model, "SVM Decision Boundary (Binary Classification)")
