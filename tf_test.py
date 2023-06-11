import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Generate demo data
np.random.seed(42)
X_train = np.random.rand(1000, 10, 1)
y_train = np.random.randint(0, 2, size=(1000,))

# Define the LSTM model
def build_lstm_model(n_units=64, optimizer='adam'):
    model = Sequential()
    model.add(LSTM(n_units, input_shape=(10, 1), dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# wrap your Keras model in a scikit-learn estimator
estimator = KerasClassifier(build_fn=build_lstm_model())

# Define the hyperparameters to be tuned
param_grid = {
    'n_units': [32, 64, 128],
    'optimizer': ['adam', 'rmsprop']
}

# Create a grid search object
grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=param_grid
)

# Fit the grid search object to the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and accuracy score
print("Best hyperparameters: ", grid_search.best_params_)
print("Best accuracy: ", grid_search.best_score_)

# Train the final LSTM model with the best hyperparameters
best_params = grid_search.best_params_
lstm_model = build_lstm_model(n_units=best_params['n_units'], dropout=best_params['dropout'], optimizer=best_params['optimizer'])
lstm_model.fit(X_train, y_train, epochs=10, batch_size=32)