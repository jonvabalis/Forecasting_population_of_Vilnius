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

models = {
    #"DecisionTreeClassifier": DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_leaf=1, min_samples_split=2),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    #"RandomForestRegressor": RandomForestRegressor(n_jobs=-1),
    "XGBRegressor": xgb.XGBRegressor(colsample_bytree=1.0, gamma=0, learning_rate=0.2, max_depth=5, n_estimators=300, reg_alpha=0, reg_lambda=0.1, subsample=0.6), #colsample_bytree=1.0, gamma=0, learning_rate=0.2, max_depth=5, n_estimators=300, reg_alpha=0, reg_lambda=0.1, subsample=0.6
    #"LGBMRegressor": lgb.LGBMRegressor(LOKY_MAX_CPU_COUNT=4),
    "CatBoostRegressor": CatBoostRegressor(verbose=False),
}
####################################################################
# calculates models' mae and saves trained models to sav files

for name, model in models.items():
    model.fit(xDataTrain, np.ravel(yDataTrain))
    y_pred = model.predict(xDataTest)
    print(name)
    print("MAE", mean_absolute_error(yDataTest, y_pred))
    print("-" * 50)
    with open(rf"C:\Users\motiejus\Downloads\ai_hackaton\{name}.sav", 'wb') as file:
        pickle.dump(model, file)

#############################################################
# opens trained models and generates an output file

for name, model in models.items():
    modelToRunTest = pickle.load(open(rf"C:\Users\motiejus\Downloads\ai_hackaton\{name}.sav", 'rb'))

    testColumnData = pd.read_csv(testPath)
    IDs = testColumnData['ID'].to_frame()

    testData = testColumnData.drop(testColumnData.columns[0], axis=1)
    resultData = pd.DataFrame({'count': modelToRunTest.predict(testData).T})

    fullData = IDs.join(resultData)
    fullData.reset_index(inplace=True, drop=True)
    fullData.to_csv(resultPath, sep=',', index=False)

####################################################################
# hyperparameter optimizing with gridsearch

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}
for name, model in models.items():
    model.fit(xDataTrain, np.ravel(yDataTrain))
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
    grid_search.fit(xDataTrain, yDataTrain)
    print("Best parameters found: ", grid_search.best_params_)
    best_clf = grid_search.best_estimator_
    y_pred = best_clf.predict(xDataTest)
    accuracy = best_clf.score(xDataTest, yDataTest)
    print("MAE: ", accuracy)
    print("-" * 50)

##########################################
# lgb model attempt

params = {
    'objective': 'regression',
    'metric': 'mse',  # Mean Squared Error
    'boosting_type': 'gbdt',  # Gradient Boosting Decision Tree
    'num_leaves': 31,  # Maximum tree leaves for base learners
    'learning_rate': 0.05,
    'feature_fraction': 0.9,  # Randomly select a fraction of features for training each tree
    'bagging_fraction': 0.8,  # Randomly select a fraction of data for training each tree
    'bagging_freq': 5,  # Perform bagging at every 5th iteration
    'verbose': 0,
    'num_iterations': 1000,  # Number of boosting iterations
    'early_stopping_rounds': 100  # Early stopping to prevent overfitting
}
train_data = lgb.Dataset(xDataTrain, label=yDataTrain)
test_data = lgb.Dataset(xDataTest, label=yDataTest)
model = lgb.train(params, train_data, valid_sets=[test_data])

y_pred = model.predict(xDataTest, num_iteration=model.best_iteration)

mse = mean_absolute_error(yDataTest, y_pred)
print("Mean Squared Error:", mse)