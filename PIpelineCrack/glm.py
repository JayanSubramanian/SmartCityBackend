import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset
data = pd.read_csv('PipelineCrack/synthetic_pipeline_crack_data.csv')

# Features and target variable
X = data.drop(['Class Label'], axis=1)
y = data['Class Label']

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Logistic Regression model
log_reg_model = LogisticRegression(random_state=2)

# 5-Fold Cross Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
cv_scores = cross_val_score(log_reg_model, X_scaled, y, cv=kf, scoring='accuracy')

# Train the model on the full dataset
log_reg_model.fit(X_scaled, y)

# Save the model and scaler for future use
joblib.dump(log_reg_model, 'logistic_reg_pipeline_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Display cross-validation results
print(f'Cross-Validation Accuracy Scores: {cv_scores}')
print(f'Mean Accuracy: {cv_scores.mean()}')
print(f'Standard Deviation: {cv_scores.std()}')
# Predictions
y_pred = log_reg_model.predict(X_scaled)

# Confusion Matrix
conf_matrix = confusion_matrix(y, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Crack', 'Crack'], yticklabels=['No Crack', 'Crack'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

# Display cross-validation results
print(f'Cross-Validation Accuracy Scores: {cv_scores}')
print(f'Mean Accuracy: {cv_scores.mean()}')
print(f'Standard Deviation: {cv_scores.std()}')
