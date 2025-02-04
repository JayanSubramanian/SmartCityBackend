import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Load the dataset
data = pd.read_csv('/Users/rash/Documents/Files/Self/College/Semester 4/Projects/SmartCityBackend/PipelineCrack/synthetic_pipeline_crack_data.csv')

# Features and target variable
X = data.drop(['Class Label'], axis=1)
y = data['Class Label']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVC model
svc_model = SVC(kernel='rbf', random_state=42)
svc_model.fit(X_train_scaled, y_train)

# Save the model and scaler for future use
joblib.dump(svc_model, 'svc_pipeline_crack_model.pkl')
joblib.dump(scaler, 'scaler.pkl')


y_pred = svc_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print Evaluation Metrics
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)
