
from ML_support_service.categorical_encoding import CategoricalEncoder

import pandas as pd

# Sample DataFrame
data = pd.DataFrame({'Category': ['A', 'B', 'C', 'A', 'B', 'C']})
encoder = CategoricalEncoder(data)
encoded_df = encoder.label_encode('Category')
print(encoded_df)


# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col
#
# # Initialize Spark Session
# spark = SparkSession.builder.appName("CategoricalEncoding").getOrCreate()
#
# # Sample DataFrame
# data = spark.createDataFrame([('A',), ('B',), ('C',), ('A',), ('B',), ('C',)], ['Category'])
# encoder = CategoricalEncoder(data)
# encoded_df = encoder.one_hot_encode('Category')
# encoded_df.show()
