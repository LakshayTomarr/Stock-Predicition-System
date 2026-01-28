import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from  pandas_datareader import data as web
import datetime as dt
import yfinance as yf 
import streamlit as st

start = dt.datetime(2010,1,1)
end = dt.datetime(2019,12,31)
df = yf.download('AAPL', start=start, end=end) 
print(df.head())

st.subheader('data from 2010 - 2019')
st.write(df.describe())
#df.info()
#df.isnull().sum()
#df.describe()
df = df.reset_index()
#df.shape
#df.head()
print(df.columns)

#candlestiks
df.to_csv("powergrid.csv")
data01 = pd.read_csv("powergrid.csv")
print(data01.head())

import plotly.graph_objects as go
fig = go.Figure(data=[go.Candlestick(x = data01['Date'], open = data01['Open'], high = data01['High'], low = data01['Low'], close = data01['Close'] )])
fig.update_layout(xaxis_rangeslider_visible = False)
fig.show()

df = df.drop(['Date'], axis = 1)
print(df.head())

plt.figure(figsize = (12 , 6))
stock = "powergrid"
plt.plot(df['Close'], label = f'{stock} Closing prise',linewidth = 1)
plt.title(f'{stock} Closing prices over time')
plt.xlabel('Date')
plt.ylabel('Closing prices (USD)')
plt.legend()
plt.show()

plt.figure(figsize = (12 , 6))
stock = "powergrid"
plt.plot(df['Open'], label = f'{stock} Opening prise',linewidth = 1)
plt.title(f'{stock} Opening prices over time')
plt.xlabel('Date')
plt.ylabel('Opening prices (USD)')
plt.legend()
plt.show()

plt.figure(figsize = (12 , 6))
stock = "powergrid"
plt.plot(df['High'], label = f'{stock} High prise',linewidth = 1)
plt.title(f'{stock} High prices over time')
plt.xlabel('Date')
plt.ylabel('High prices (USD)')
plt.legend()
plt.show()

plt.figure(figsize = (12 , 6))
stock = "powergrid"
plt.plot(df['Volume'], label = f'{stock} Volume',linewidth = 1)
plt.title(f'{stock} Volume over time')
plt.legend()
plt.show()


# Moving average

temp_data = [10,20,30,40,50,60,70,80,90]
print(sum(temp_data[1:6])/5)

import pandas as pd 
df01 = pd.DataFrame(temp_data)
df01.rolling(5).mean()

ma100 = df.Close.rolling(100).mean()
print(ma100)
ma200 = df.Close.rolling(200).mean()
print(ma200)

plt.figure(figsize = (12 , 6))
stock = "powergrid"
plt.plot(df.Close, label = f'{stock} Closing price',linewidth = 1)
plt.plot(ma100, label = f'{stock} Moving Average 100',linewidth = 1)
plt.plot(ma200, label = f'{stock} Moving Average 200',linewidth = 1)
plt.legend()
plt.show()

ema100 = df.Close.ewm(span = 100 , adjust = False).mean()
ema200 = df['Close'].ewm(span = 200 , adjust = False).mean()
plt.figure(figsize = (12 , 6))
stock = "powergrid"
plt.plot(df.Close, label = f'{stock} Closing price',linewidth = 1)
plt.plot(ema100, label = f'{stock} Exp. Moving Average 100',linewidth = 1)
plt.plot(ema200, label = f'{stock} Exp. Moving Average 200',linewidth = 1)
plt.legend()
plt.show()
#Data training and testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)
data_training_array.shape

x_train =[]
y_train =[]

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i])

x_train, y_train = np.array(x_train) , np.array(y_train)
x_train.shape

# Model Byuilding 

from tensorflow.keras.layers import Dense,Dropout,LSTM
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.4))

model.add(LSTM(units = 120, activation = 'relu', return_sequences = True))
model.add(Dropout(0.5))

model.add(Dense(units = 1))
model.summary()

model.compile(optimizer = 'adam' , loss = 'mean_squared_error')
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("first five samples of y_train:", y_train[:5])
model.fit(x_train, y_train, epochs = 50)

past_100_days = data_training.tail(100)
final_df =pd.concat([past_100_days,data_testing], ignore_index = True)
print(final_df.head())
input_data = scaler.fit_transform(final_df)

x_test =[]
y_test =[]

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test) , np.array(y_test)
x_test.shape

y_predicted = model.predict(x_test)
y_predicted.shape

scaler.scale_

scaler_factor = 1 / 0.003516
y_predicted = y_predicted * scaler_factor
y_test = y_test * scaler_factor

plt.figure(figsize = (12 , 6))
stock = "powergrid"
plt.plot(y_test.flatten(), label='original price')
plt.plot(y_predicted.flatten(), label = 'Predicted price')
plt.legend()
plt.show()
