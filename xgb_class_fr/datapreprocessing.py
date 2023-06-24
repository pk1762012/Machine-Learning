import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from scipy.stats import chi2_contingency

class DataPreprocessing:
    def __init__(self, data):
        self.data = data
    
    def create_categories(self, col, num_bins=10):
        """
        Create automated categories/bins for a given column.
        Outliers are assigned to one category, and missing values are assigned another category.
        """
        # Handle missing values
        self.data[col+'_missing'] = np.where(self.data[col].isnull(), 1, 0)
        self.data[col].fillna(-999, inplace=True)
        
        # Handle outliers
        q1 = self.data[col].quantile(0.25)
        q3 = self.data[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5*iqr
        upper_bound = q3 + 1.5*iqr
        self.data[col+'_outlier'] = np.where((self.data[col] < lower_bound) | (self.data[col] > upper_bound), 1, 0)
        self.data[col] = np.where((self.data[col] < lower_bound) | (self.data[col] > upper_bound), -999, self.data[col])
        
        # Create categories
        self.data[col+'_bin'] = pd.qcut(self.data[col], num_bins, duplicates='drop')
        self.data[col+'_bin'] = self.data[col+'_bin'].astype(str)
        
    def calculate_woe(self, col, target):
        """
        Calculate WOE values for each category of a given column.
        """
        total_good = self.data[target].value_counts()[0]
        total_bad = self.data[target].value_counts()[1]
        grouped = self.data.groupby(col+'_bin').agg({target: ['count', 'sum']})
        grouped.columns = ['total', 'bad']
        grouped['good'] = grouped['total'] - grouped['bad']
        grouped['woe'] = np.log((grouped['good']/total_good) / (grouped['bad']/total_bad))
        return grouped['woe'].to_dict()
    
    def calculate_iv(self, col, target):
        """
        Calculate Information Value for a given column.
        """
        total_good = self.data[target].value_counts()[0]
        total_bad = self.data[target].value_counts()[1]
        grouped = self.data.groupby(col+'_bin').agg({target: ['count', 'sum']})
        grouped.columns = ['total', 'bad']
        grouped['good'] = grouped['total'] - grouped['bad']
        grouped['woe'] = np.log((grouped['good']/total_good) / (grouped['bad']/total_bad))
        grouped['iv'] = (grouped['good']/total_good - grouped['bad']/total_bad) * grouped['woe']
        return grouped['iv'].sum()
    
    def select_top_variables(self, target, num_vars=100):
        """
        Select top variables based on Information Value.
        """
        iv = []
        for col in self.data.columns:
            if col != target:
                iv.append((col, self.calculate_iv(col, target)))
        iv_df = pd.DataFrame(iv, columns=['Variable', 'IV'])
        iv_df = iv_df.sort_values('IV', ascending=False).reset_index(drop=True)
        top_vars = iv_df['Variable'][:num_vars]
        self.data = self.data[top_vars.append(pd.Index([target]))]
    
    def split_data(self, target, test_size=0.3):
        """
        Split data into train and test sets.
        """
        X = self.data.drop(target, axis=1)
        y = self.data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test
