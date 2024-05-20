import pathlib
import numpy as np
import pandas as pd
from keras.layers import LSTM

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

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


########################################################
#xDataTrainReshaped = xDataTrain.values.reshape(xDataTrain.shape[0], 1, 4)
print("Shape of the input tensor:")
print(xDataTrain.shape)

model = Sequential()
model.add(Dense(1024, input_shape=(4,), activation='relu'))
model.add(BatchNormalization())

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())

# Adding dense layers
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])

# Train the model
model.fit(xDataTrain, yDataTrain, epochs=100, batch_size=32, validation_split=0.2)
#y_pred = model.predict(xDataTest)

print("-" * 50)
print("MAE: ", model.evaluate(xDataTest, yDataTest)[0])
print("-" * 50)



testColumnData = pd.read_csv(testPath)
IDs = testColumnData['ID'].to_frame()
testData = testColumnData.drop(testColumnData.columns[0], axis=1)
resultData = pd.DataFrame({'count': model.predict(testData).flatten()})
fullData = IDs.join(resultData)
fullData.reset_index(inplace=True, drop=True)
fullData.to_csv(resultPath, sep=',', index=False)