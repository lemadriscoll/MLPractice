import pandas
import matplotlib.pyplot as plt

# download data-set and plot
dataframe1 = pandas.read_csv('/Users/arianalemadriscoll/Documents/SeasonalityData/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
plt.plot(dataframe1)
# plt.show()

# Import SciPy packages
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import math
import sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import  mean_squared_error

# Fix random seed for reproducibility
numpy.random.seed(7)

# Load the dataset
dataframe1 = pd.read_csv('/Users/arianalemadriscoll/Documents/SeasonalityData/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset1 = dataframe1.values
dataset1 = dataset1.astype('float32')

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset1 = scaler.fit_transform(dataset1)

# Split into train and test sets
train_size = int(len(dataset1)*0.67)
test_size = len(dataset1) - train_size
train, test = dataset1[0:train_size, :], dataset1[train_size:len(dataset1), :]
print(len(train), len(test))

# Convert an array of values into a data-set matrix
def create_dataset(dataset1, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset1)-look_back-1):
        a = dataset1[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset1[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

# Reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# Invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# Shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset1)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# Shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset1)-1, :] = testPredict

# Plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

# Plot baseline and predictions