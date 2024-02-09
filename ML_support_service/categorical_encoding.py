import pandas as pd
from sklearn.preprocessing import LabelEncoder

try:
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.ml.feature import StringIndexer, OneHotEncoder
    pyspark_available = True
except ImportError:
    pyspark_available = False
    SparkDataFrame = None  # Placeholder for type checking without PySpark

class CategoricalEncoder:
    """
    A class for encoding categorical data in both Pandas and Spark DataFrames.

    This class supports Label Encoding and One-Hot Encoding. It automatically detects
    the type of DataFrame (Pandas or Spark) and applies the encoding accordingly.

    Attributes:
    -----------
    df : pd.DataFrame or SparkDataFrame
        The DataFrame containing the data to be encoded.
    is_spark_df : bool
        A marker to indicate whether the DataFrame is a Spark DataFrame.

    Methods:
    --------
    __init__(self, df):
        Initializes the CategoricalEncoder with a DataFrame.
    label_encode(self, column_name: str):
        Applies Label Encoding to a specified column.
    one_hot_encode(self, column_name: str):
        Applies One-Hot Encoding to a specified column.
    """

    def __init__(self, df):
        """
        Initializes the CategoricalEncoder with a DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame or SparkDataFrame
            The DataFrame with the data to encode.
        """
        self.df = df
        self.is_spark_df = isinstance(df, SparkDataFrame)

    def label_encode(self, column_name: str):
        """
        Applies Label Encoding to a specified column.

        For Pandas DataFrame, it uses sklearn's LabelEncoder.
        For Spark DataFrame, it uses PySpark's StringIndexer.

        Parameters:
        -----------
        column_name : str
            The name of the column to apply Label Encoding.

        Returns:
        --------
        The DataFrame with the specified column label encoded.
        """
        if self.is_spark_df:
            if not pyspark_available:
                raise ImportError("PySpark is not available in the environment.")
            indexer = StringIndexer(inputCol=column_name, outputCol=f"{column_name}_indexed")
            self.df = indexer.fit(self.df).transform(self.df)
        else:
            le = LabelEncoder()
            self.df[column_name] = le.fit_transform(self.df[column_name])
        return self.df

    def one_hot_encode(self, column_name: str):
        """
        Applies One-Hot Encoding to a specified column.

        For Pandas DataFrame, it uses pandas.get_dummies.
        For Spark DataFrame, it uses PySpark's OneHotEncoder after StringIndexing.

        Parameters:
        -----------
        column_name : str
            The name of the column to apply One-Hot Encoding.

        Returns:
        --------
        The DataFrame with the specified column one-hot encoded.
        """
        if self.is_spark_df:
            if not pyspark_available:
                raise ImportError("PySpark is not available in the environment.")
            indexer = StringIndexer(inputCol=column_name, outputCol=f"{column_name}_indexed")
            indexed = indexer.fit(self.df).transform(self.df)
            encoder = OneHotEncoder(inputCols=[f"{column_name}_indexed"], outputCols=[f"{column_name}_ohe"])
            self.df = encoder.fit(indexed).transform(indexed)
        else:
            dummies = pd.get_dummies(self.df[column_name], prefix=column_name)
            self.df = pd.concat([self.df, dummies], axis=1).drop(column_name, axis=1)
        return self.df



import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

try:
    import pyspark.sql.functions as F
    from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, Pipeline
except ImportError:
    print("PySpark not found. Spark-specific functionality will be unavailable.")

class CategoricalEncoder_v2:
    """
    A class for encoding categorical data in various ways.

    Attributes:
        df_type (str): Indicates the type of DataFrame used ('pandas' or 'spark').
    """

    def __init__(self, df_type='pandas'):
        """
        Initializes the CategoricalEncoder object.

        Args:
            df_type (str): The type of DataFrame to be used.
                Options are 'pandas' or 'spark' (Default: 'pandas').
        """
        self.df_type = df_type

    def _label_encoding(self, X):
        """Performs label encoding on categorical columns."""

        if self.df_type == 'pandas':
            le = LabelEncoder()
            for col in X.select_dtypes(include='object'):
                X[col] = le.fit_transform(X[col].astype(str))

        elif self.df_type == 'spark':
            for col in X.select_dtypes(include='string').columns:
                indexer = StringIndexer(inputCol=col, outputCol=col + "_index")
                X = indexer.fit(X).transform(X)

        else:
            raise ValueError("Invalid df_type. Use 'pandas' or 'spark'.")

        return X

    def _one_hot_encoding(self, X):
        """Performs one-hot encoding on categorical columns."""

        if self.df_type == 'pandas':
            ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
            encoded_cols = pd.DataFrame(ohe.fit_transform(X[X.select_dtypes(include='object').columns]))
            encoded_cols.index = X.index
            X = X.drop(X.select_dtypes(include='object').columns, axis=1)
            X = pd.concat([X, encoded_cols], axis=1)

        elif self.df_type == 'spark':
            categorical_cols = [col for col, dtype in X.dtypes if dtype == 'string']
            indexers = [StringIndexer(inputCol=c, outputCol=c + "_index") for c in categorical_cols]
            encoder = OneHotEncoderEstimator(
                inputCols=[c + "_index" for c in categorical_cols],
                outputCols=[c + "_vec" for c in categorical_cols]
            )
            pipeline = Pipeline(stages=indexers + [encoder])
            X = pipeline.fit(X).transform(X)

        else:
            raise ValueError("Invalid df_type. Use 'pandas' or 'spark'.")

        return X

    # Add methods for other encoding techniques (e.g., _ordinal_encoding)

    def fit(self, X):
        """Fits the selected encoder to the categorical data. Not always needed."""
        pass  # Implement if fitting is required for certain encoding methods

    def transform(self, X):
        """Transforms the categorical data using a chosen encoding method."""
        # Example usage - call a specific encoding method
        return self._one_hot_encoding(X.copy())

    def fit_transform(self, X):
        """Fits the encoder to the data and then transforms it."""
        self.fit(X)
        return self.transform(X)
