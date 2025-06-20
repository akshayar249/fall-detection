import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


# Load the dataset
df = pd.read_csv('newmergeddataset.csv')

# Define the window and step size
window_size = 80
step_size = 20

# List to store features and labels
features = []
labels = []

# Loop through the dataset and extract features
for i in range(0, len(df) - window_size, step_size):
    window_data = df.iloc[i:i+window_size]

    # Use Pandas mode() for activity label
    activity_label = window_data['activity'].mode()[0]  

    # Extract sensor data
    accel_x, accel_y, accel_z = window_data['accel_x'], window_data['accel_y'], window_data['accel_z']
    gyro_x, gyro_y, gyro_z = window_data['gyro_x'], window_data['gyro_y'], window_data['gyro_z']

    # Compute statistical features
    window_features = [
        np.mean(accel_x), np.mean(accel_y), np.mean(accel_z),
        np.mean(gyro_x), np.mean(gyro_y), np.mean(gyro_z),
        np.std(accel_x), np.std(accel_y), np.std(accel_z),
        np.std(gyro_x), np.std(gyro_y), np.std(gyro_z),
        skew(accel_x), skew(accel_y), skew(accel_z),
        skew(gyro_x), skew(gyro_y), skew(gyro_z),
        kurtosis(accel_x), kurtosis(accel_y), kurtosis(accel_z),
        kurtosis(gyro_x), kurtosis(gyro_y), kurtosis(gyro_z),
        np.max(accel_x), np.max(accel_y), np.max(accel_z),
        np.max(gyro_x), np.max(gyro_y), np.max(gyro_z),
        np.min(accel_x), np.min(accel_y), np.min(accel_z),
        np.min(gyro_x), np.min(gyro_y), np.min(gyro_z)
    ]

    # Append features and labels
    features.append(window_features)
    labels.append(activity_label)

# Convert features list into DataFrame
features_df = pd.DataFrame(features, columns=[
    'mean_accel_x', 'mean_accel_y', 'mean_accel_z', 'mean_gyro_x', 'mean_gyro_y', 'mean_gyro_z',
    'std_accel_x', 'std_accel_y', 'std_accel_z', 'std_gyro_x', 'std_gyro_y', 'std_gyro_z',
    'skew_accel_x', 'skew_accel_y', 'skew_accel_z', 'skew_gyro_x', 'skew_gyro_y', 'skew_gyro_z',
    'kurtosis_accel_x', 'kurtosis_accel_y', 'kurtosis_accel_z', 'kurtosis_gyro_x', 'kurtosis_gyro_y', 'kurtosis_gyro_z',
    'max_accel_x', 'max_accel_y', 'max_accel_z', 'max_gyro_x', 'max_gyro_y', 'max_gyro_z',
    'min_accel_x', 'min_accel_y', 'min_accel_z', 'min_gyro_x', 'min_gyro_y', 'min_gyro_z'
])

# Add activity label column
features_df['activity'] = labels

# Save the final dataset
features_df.to_csv('labeled_sensor_data_with_skewness_kurtosis1.csv', index=False)
import pandas as pd

# Load the dataset with window-based labeled activities
df = pd.read_csv('labeled_sensor_data_with_skewness_kurtosis1.csv')
print(features_df.head())
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('labeled_sensor_data_with_skewness_kurtosis1.csv')

# Assume last column is the target
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Labels

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for future use
import joblib
joblib.dump(scaler, "scaler.pkl")

import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Define range of K values
k_values = list(range(1, 21))  # K from 1 to 20
cv_scores = []  # Store cross-validation scores

# Time-Series Aware Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)  # Rolling time-series split

# Perform CV for each K
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=tscv, scoring='accuracy')
    cv_scores.append(scores.mean())  # Store mean accuracy

# Best K selection
best_k = k_values[np.argmax(cv_scores)]
best_accuracy = max(cv_scores)

print(f"🔹 Best K: {best_k}, Best Accuracy: {best_accuracy:.4f}")

# Plot K vs Accuracy
plt.figure(figsize=(8, 5))
plt.plot(k_values, cv_scores, marker='o', linestyle='-', color='b')
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Cross-Validation Accuracy")
plt.title("KNN Hyperparameter Tuning: K vs Accuracy")
plt.xticks(k_values)
plt.grid(True)
plt.show()

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define hyperparameter grid for SVM
svm_param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'gamma': ['scale', 0.01, 0.1, 1],  # Kernel coefficient
    'kernel': ['rbf']  # Using RBF kernel
}

# Define hyperparameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees
    'max_depth': [None, 10, 20]  # Maximum depth of trees
}


# Tune SVM
svm = SVC(probability=True)  # Enable probability estimates
svm_grid_search = GridSearchCV(svm, svm_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
svm_grid_search.fit(X_train_scaled, y_train)

# Tune Random Forest
rf = RandomForestClassifier(random_state=42)
rf_grid_search = GridSearchCV(rf, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid_search.fit(X_train_scaled, y_train)

# Get best parameters
best_svm_params = svm_grid_search.best_params_
best_rf_params = rf_grid_search.best_params_

# Print best parameters
print(f"🔹 Best SVM Parameters: {best_svm_params}")
print(f"🔹 Best Random Forest Parameters: {best_rf_params}") 

# Get best models
best_svm_model = svm_grid_search.best_estimator_
best_rf_model = rf_grid_search.best_estimator_

# Evaluate best models on test data
svm_test_acc = best_svm_model.score(X_test_scaled, y_test)
rf_test_acc = best_rf_model.score(X_test_scaled, y_test)

print(f"✅ Best SVM Test Accuracy: {svm_test_acc:.4f}")
print(f"✅ Best Random Forest Test Accuracy: {rf_test_acc:.4f}")
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib



# Define Base Models with Best Hyperparameters
base_models = [
    ('knn', KNeighborsClassifier(n_neighbors=2)),  # Best K
    ('svm', SVC(C=10, gamma='scale', kernel='rbf', probability=True)),  # Best SVM
    ('logreg', LogisticRegression())  # Logistic Regression
]

# Define Stacking Classifier with Random Forest as Final Model
stack_model = StackingClassifier(estimators=base_models, final_estimator=RandomForestClassifier(max_depth=None, n_estimators=50, random_state=42))

# Train the Stacking Model
stack_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = stack_model.predict(X_test_scaled)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print results
print(f"✅ Stacked Model Accuracy: {accuracy:.4f}")
print("\n🔹 Classification Report:\n", class_report)
print("\n🔹 Confusion Matrix:\n", conf_matrix)

# Save the trained stacking model
joblib.dump(stack_model, "fall_detection_stacked_model.pkl")
import numpy as np
import pandas as pd
import joblib
from scipy.stats import skew, kurtosis

# Load trained stacking model & scaler
model = joblib.load("fall_detection_stacked_model.pkl")
scaler = joblib.load("scaler.pkl")

# Feature names
feature_names = [
    'mean_accel_x', 'mean_accel_y', 'mean_accel_z', 'mean_gyro_x', 'mean_gyro_y', 'mean_gyro_z',
    'std_accel_x', 'std_accel_y', 'std_accel_z', 'std_gyro_x', 'std_gyro_y', 'std_gyro_z',
    'skew_accel_x', 'skew_accel_y', 'skew_accel_z', 'skew_gyro_x', 'skew_gyro_y', 'skew_gyro_z',
    'kurtosis_accel_x', 'kurtosis_accel_y', 'kurtosis_accel_z', 'kurtosis_gyro_x', 'kurtosis_gyro_y', 'kurtosis_gyro_z',
    'max_accel_x', 'max_accel_y', 'max_accel_z', 'max_gyro_x', 'max_gyro_y', 'max_gyro_z',
    'min_accel_x', 'min_accel_y', 'min_accel_z', 'min_gyro_x', 'min_gyro_y', 'min_gyro_z'
]

def compute_features(accel_data, gyro_data):
    """Compute statistical features from accelerometer and gyroscope readings."""
    features = [
        np.mean(accel_data, axis=0), np.mean(gyro_data, axis=0),
        np.std(accel_data, axis=0), np.std(gyro_data, axis=0),
        skew(accel_data, axis=0), skew(gyro_data, axis=0),
        kurtosis(accel_data, axis=0), kurtosis(gyro_data, axis=0),
        np.max(accel_data, axis=0), np.max(gyro_data, axis=0),
        np.min(accel_data, axis=0), np.min(gyro_data, axis=0)
    ]
    return np.hstack(features)  # Flatten into a 1D array

def predict_fall(accel_data, gyro_data):
    """Predict if the given sensor data indicates a fall or not."""
    features = compute_features(accel_data, gyro_data)
    
    # Convert to DataFrame to maintain feature names
    features_df = pd.DataFrame([features], columns=feature_names)
    
    # Scale features
    features_scaled = scaler.transform(features_df)
    
    # Predict
    prediction = model.predict(features_scaled)[0]  
    return "Fall" if prediction == 1 else "Not Fall"

# Simulated Real-Time Sensor Data (Replace with actual sensor readings)
accel_example = np.random.rand(80, 3)  # Replace with real accelerometer data
gyro_example = np.random.rand(80, 3)   # Replace with real gyroscope data

# Make Prediction
result = predict_fall(accel_example, gyro_example)
print("🔹 Predicted Activity:", result)

