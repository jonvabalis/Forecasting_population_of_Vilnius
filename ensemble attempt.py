import pathlib
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.svm import SVR

trainPath = pathlib.Path(r"C:\Users\motiejus\Downloads\ai_hackaton\train.csv")
testPath = pathlib.Path(r"C:\Users\motiejus\Downloads\ai_hackaton\test.csv")
resultPath = pathlib.Path(r'C:\Users\motiejus\Downloads\ai_hackaton\output.csv')

xColumnName = ['district_id', 'age_bin_id', 'gender_id', 'as_of_date_id']

yColumnName = ['count']

trainColumnData = pd.read_csv(trainPath)
trainColumnData = trainColumnData.dropna()

# testColumnData = pd.read_csv(testPath)

x_dataTrain = trainColumnData[xColumnName]
y_dataTrain = trainColumnData[yColumnName]

xDataTrain, xDataTest, yDataTrain, yDataTest = train_test_split(x_dataTrain, y_dataTrain, test_size=0.2) #random_state=42

#xDataTrain, xDataTest = train_test_split(x_dataTrain, y_dataTrain, test_size=0.2, random_state=42)

models = {
    #"DecisionTreeClassifier": DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_leaf=1, min_samples_split=2),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    #"RandomForestRegressor": RandomForestRegressor(n_jobs=-1),
    "XGBRegressor": xgb.XGBRegressor(colsample_bytree=1.0, gamma=0, learning_rate=0.2, max_depth=5, n_estimators=300, reg_alpha=0, reg_lambda=0.1, subsample=0.6), #colsample_bytree=1.0, gamma=0, learning_rate=0.2, max_depth=5, n_estimators=300, reg_alpha=0, reg_lambda=0.1, subsample=0.6
    #"LGBMRegressor": lgb.LGBMRegressor(LOKY_MAX_CPU_COUNT=4),
    "CatBoostRegressor": CatBoostRegressor(verbose=False),
}
paramsold = {
    'objective': 'regression',
    'metric': 'mae',  # Mean Squared Error
    'boosting_type': 'gbdt',  # Gradient Boosting Decision Tree
    'num_leaves': 275,  # Maximum tree leaves for base learners
    'learning_rate': 0.05,
    'feature_fraction': 0.9,  # Randomly select a fraction of features for training each tree
    'bagging_fraction': 0.8,  # Randomly select a fraction of data for training each tree
    'bagging_freq': 5,  # Perform bagging at every 5th iteration
    'verbose': 0,
    'num_iterations': 5000,  # Number of boosting iterations
}
params = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'num_leaves': 250,  # Increasing num_leaves further to capture more complex patterns
    'learning_rate': 0.01,  # Decrease learning rate for smoother convergence and finer adjustments
    'feature_fraction': 0.6,  # Further reduce feature fraction to promote better generalization
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'num_iterations': 3000,  # Increase the number of boosting iterations for more comprehensive learning
    'early_stopping_rounds': 100,
    'min_data_in_leaf': 50,  # Increase min_data_in_leaf for more robustness against overfitting
    'lambda_l1': 0.05,  # Increase L1 regularization term for more feature selection
    'lambda_l2': 0.1,  # Keep L2 regularization term for robustness
    'min_gain_to_split': 0.2,  # Increase min_gain_to_split for more conservative splitting
    'max_depth': 10,  # Set a maximum depth to control the complexity of individual trees
    'bagging_seed': 42,  # Set bagging seed for reproducibility
    'random_state': 42  # Set random state for reproducibility
}
dt_model = DecisionTreeRegressor()
dt_model.fit(xDataTrain, np.ravel(yDataTrain))
dt_pred = dt_model.predict(xDataTest)

xgb_model = xgb.XGBRegressor(colsample_bytree=1.0, gamma=0, learning_rate=0.2, max_depth=5, n_estimators=300, reg_alpha=0, reg_lambda=0.1, subsample=0.6)
xgb_model.fit(xDataTrain, np.ravel(yDataTrain))
xgb_pred = xgb_model.predict(xDataTest)

train_data = lgb.Dataset(xDataTrain, label=yDataTrain)
test_data = lgb.Dataset(xDataTest, label=yDataTest)
model = lgb.train(params, train_data, valid_sets=[test_data])

lgb_pred = model.predict(xDataTest, num_iteration=model.best_iteration)


ens_pred = (dt_pred + xgb_pred+ lgb_pred) / 3

mse = mean_absolute_error(yDataTest, lgb_pred)
print("mean_absolute_error", mse)

for name, model in models.items():
    model.fit(xDataTrain, np.ravel(yDataTrain))
    y_pred = model.predict(xDataTest)
    print(name)
    #print(confusion_matrix(yDataTest, y_pred))
    #print(classification_report(yDataTest, y_pred))
    #print("Accuracy", accuracy_score(yDataTest, y_pred))
    print("MEA", mean_absolute_error(yDataTest, y_pred))
    #print(multilabel_confusion_matrix(yDataTest, y_pred))
    print("-" * 50)

