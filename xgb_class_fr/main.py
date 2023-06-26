import pandas as pd
from datapreprocessing import DataPreprocessing
from xgboostmodel import XGBoostModel
import joblib

def main():
    # Load data
    data = pd.read_csv('creditcard.csv')
    columns_remove = pd.Series('Time')
    data = data.drop(columns=columns_remove)
    data_columns = data.columns
    target = 'Class'
    # Preprocess data
    dp = DataPreprocessing(data, data_columns)
    for col in data.columns:
        if data[col].dtype == 'object':
            dp.create_categories(col, target, num_bins=10)
        else:
            dp.create_categories(col, target, num_bins=20)
    dp.select_top_variables(target, num_vars=100)
    X_train, X_test, y_train, y_test = dp.split_data('Class')
    
    # Build and tune XGBoost model
    xgbm = XGBoostModel(X_train, y_train, X_test, y_test)
    xgbm.hyperparameter_tuning()
    xgbm.train_model()
    auc, tn, fp, fn, tp, tdr = xgbm.evaluate_model()
    
    # Save model
    joblib.dump(xgbm.model, 'model.pkl')
    
if __name__ == '__main__':
    main()
