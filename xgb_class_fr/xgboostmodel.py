import xgboost as xgb
from sklearn.metrics import roc_auc_score, confusion_matrix

class XGBoostModel:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    
    def hyperparameter_tuning(self):
        """
        Tune hyperparameters of XGBoost model using Bayesian Optimization.
        """
        def xgb_evaluate(max_depth, gamma, colsample_bytree):
            params = {'eval_metric': 'auc',
                      'max_depth': int(max_depth),
                      'subsample': 0.8,
                      'eta': 0.1,
                      'gamma': gamma,
                      'colsample_bytree': colsample_bytree}
            cv_result = xgb.cv(params, xgb.DMatrix(self.X_train, label=self.y_train, enable_categorical=True), num_boost_round=1000, nfold=5)
            return cv_result['test-auc-mean'].iloc[-1]
        
        from bayes_opt import BayesianOptimization
        xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 7),
                                                     'gamma': (0, 1),
                                                     'colsample_bytree': (0.3, 0.9)})
        xgb_bo.maximize(init_points=3, n_iter=10)
        params = xgb_bo.max['params']
        params['max_depth'] = int(params['max_depth'])
        self.params = params
    
    def train_model(self):
        """
        Train XGBoost model with GPU mode.
        """
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dtest = xgb.DMatrix(self.X_test, label=self.y_test)
        self.params['tree_method'] = 'gpu_hist'
        self.params['predictor'] = 'gpu_predictor'
        self.model = xgb.train(self.params, dtrain, num_boost_round=1000, evals=[(dtest, 'test')], early_stopping_rounds=10)
    
    def evaluate_model(self):
        """
        Evaluate XGBoost model using AUC, confusion matrix, and TDR.
        """
        y_pred = self.model.predict(xgb.DMatrix(self.X_test))
        auc = roc_auc_score(self.y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred.round()).ravel()
        tdr = tp / (tp + fn)
        return auc, tn, fp, fn, tp, tdr
