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

xDataTrain, xDataTest, yDataTrain, yDataTest = train_test_split(x_dataTrain, y_dataTrain, test_size=0.2, random_state=42)

models = {
    #"DecisionTreeClassifier": DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_leaf=1, min_samples_split=2),
    #"DecisionTreeRegressor": DecisionTreeRegressor(),
    #"RandomForestRegressor": RandomForestRegressor(n_jobs=-1),
    "XGBRegressor": xgb.XGBRegressor(), #colsample_bytree=1.0, gamma=0, learning_rate=0.2, max_depth=5, n_estimators=300, reg_alpha=0, reg_lambda=0.1, subsample=0.6
    #"LGBMRegressor": lgb.LGBMRegressor(LOKY_MAX_CPU_COUNT=4),
    #"CatBoostRegressor": CatBoostRegressor(verbose=False),
}

####################################################################
# hyperparameter optimizing

param_grid = {
    'n_estimators': [100, 200, 300],  # Number of boosting rounds
    'learning_rate': [0.05, 0.1, 0.2],  # Step size shrinkage
    'max_depth': [3, 4, 5],  # Maximum depth of a tree
    'subsample': [0.6, 0.8, 1.0],  # Subsample ratio of the training instance
    'colsample_bytree': [0.6, 0.8, 1.0],  # Subsample ratio of columns when constructing each tree
    'gamma': [0, 0.1, 0.2],  # Minimum loss reduction required to make a further partition on a leaf node
    'reg_alpha': [0, 0.1, 0.5],  # L1 regularization term on weights
    'reg_lambda': [0, 0.1, 0.5]  # L2 regularization term on weights
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

####################################################################

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
    with open(rf"C:\Users\motiejus\Downloads\ai_hackaton\{name}.sav", 'wb') as file:
        pickle.dump(model, file)


#############################################################

for name, model in models.items():
    #model.fit(xDataTrain, np.ravel(yDataTrain))
    modelToRunTest = pickle.load(open(rf"C:\Users\motiejus\Downloads\ai_hackaton\{name}.sav", 'rb'))

    testColumnData = pd.read_csv(testPath)
    IDs = testColumnData['ID'].to_frame()

    testData = testColumnData.drop(testColumnData.columns[0], axis=1)
    resultData = pd.DataFrame({'count': modelToRunTest.predict(testData).T})

    fullData = IDs.join(resultData)
    fullData.reset_index(inplace=True, drop=True)
    fullData.to_csv(resultPath, sep=',', index=False)