import numpy as np
import pandas as pd
# import tensorflow as tf
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.datasets import fetch_openml
from xgboost import XGBRegressor
from cuml.metrics import mean_squared_error

from sklearn.metrics import accuracy_score, confusion_matrix
from cuml.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Load the dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Combine train and test data for preprocessing
data = pd.concat([train_data, test_data], axis=0)

# Select features for training the model (e.g., numerical features)
selected_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt', 'SalePrice']
data = data[selected_features]

# Drop rows with missing target values (SalePrice)
data = data.dropna(subset=['SalePrice'])

# Split the data into features and target
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

time1 = time.time()

# Create the Random Forest classifier
rf_model = XGBRegressor(learning_rate=0.02, n_estimators=1000, objective='reg:squarederror', nthread=6,
                         # tree_method='gpu_hist',   # It is taking longer so removed.
                         eval_metric='auc')

# Define the parameter grid for hyperparameter tuning
param_grid = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.02, 0.05]
        }

# Perform grid search with cross-validation
grid_search = RandomizedSearchCV(rf_model, param_grid, cv=3)
param_comb = 1
random_search = RandomizedSearchCV(grid_search, param_distributions=param_grid, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=3, verbose=3, random_state=1001 )
print(time.time()-time1)
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Predict on the test set
y_pred = best_model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(time.time() - time1)


print("Mean Squared Error:", mse)
# # Calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Best Accuracy:", grid_search.best_score_)
# print("Test Accuracy:", accuracy)
# print("Best Parameters:", grid_search.best_params_)
#
# # Generate confusion matrix
# cm = confusion_matrix(y_test, y_pred)
#
# print(time.time() - time1)
#
# # Plot confusion matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix')
# plt.show()
