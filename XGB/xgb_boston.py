from xgboost import XGBClassifier as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the cost function options
cost_functions = {
    'multi:softmax': 'mlogloss',
    'multi:softprob': 'mlogloss',
}

# Iterate over the cost function options
for cost_name, cost_func in cost_functions.items():
    # Define the XGBoost parameters
    params = {
        'objective': cost_func,
        'num_class': len(set(y)),
        'eval_metric': cost_name,
        'seed': 42
    }

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    # Train the XGBoost model
    model = xgb.train(params, dtrain)

    # Make predictions on the test set
    y_pred = model.predict(dtest)

    # Convert predicted probabilities to class labels
    y_pred_labels = y_pred.argmax(axis=1)

    # Evaluate the model using accuracy
    accuracy = accuracy_score(y_test, y_pred_labels)

    # Print the evaluation results
    print(f"Cost function: {cost_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print("------------------------------------")
