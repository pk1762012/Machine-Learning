import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('your_dataset.csv')

# Separate the features and target variable
X = data.drop('target', axis=1)
y = data['target']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform label encoding on the target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Create a copy of the training set for target encoding
X_train_encoded = X_train.copy()

# Iterate over categorical columns to perform target encoding
for col in X_train.select_dtypes(include='object'):
    target_mean = X_train.groupby(col)[y_train].mean()
    X_train_encoded[col] = X_train[col].map(target_mean)

# Display the encoded dataset
print(X_train_encoded.head())
