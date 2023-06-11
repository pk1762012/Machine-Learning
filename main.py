import pandas as pd
import cuml
from sklearn.model_selection import train_test_split
from cuml.preprocessing import MinMaxScaler

# Load the dataset
train_data = pd.read_csv('files_saved/train.csv')
test_data = pd.read_csv('files_saved/test.csv')

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

# Perform feature scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Reshape the input data for LSTM
window_size = 10

def create_sequences(data, y_x, window_size):
    X = []
    y = []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(y_x[i + window_size])  # Modified the indexing
    return cuml.DataFrame(X), cuml.DataFrame(y)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, window_size)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, window_size)


from cuml import LSTM
from cuml.metrics import mean_squared_error

# Build the LSTM model
model = LSTM(n_cells=128, n_layers=1, input_size=X_train_seq.shape[1], output_size=1)

# Fit the model
model.fit(X_train_seq, y_train_seq, n_epochs=10, batch_size=32)

# Make predictions
y_pred_seq = model.predict(X_test_seq)

# Calculate the mean squared error
mse = mean_squared_error(y_test_seq, y_pred_seq)

print("Mean Squared Error:", mse)
