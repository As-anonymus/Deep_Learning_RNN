#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


df = pd.read_csv(r"C:\Users\Aditya Singh\Downloads\MRTSSM448USN.csv",parse_dates=True,index_col='DATE')


# In[6]:


df.info()


# In[8]:


df.columns = ['Sales']


# In[9]:


df.plot()


# In[10]:


len(df)


# In[11]:


len(df)-18


# In[12]:


test_size = 18
test_ind = len(df) - test_size


# In[13]:


train = df.iloc[:test_ind]
test = df.iloc[test_ind:]


# In[14]:


train


# In[15]:


test


# In[16]:


from sklearn.preprocessing import MinMaxScaler


# In[17]:


scalar = MinMaxScaler()


# In[18]:


scalar.fit(train)


# In[19]:


scaled_train = scalar.transform(train
                               )


# In[20]:


scaled_test = scalar.transform(test
                               )


# In[21]:


from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


# In[22]:


len(test)


# In[23]:


length = 12
generator = TimeseriesGenerator(scaled_train,scaled_train,length=length,batch_size=1)


# In[24]:


X,y = generator[0]


# In[28]:


len(X[0])


# In[26]:


y


# In[27]:


scaled_train


# In[29]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[30]:


n_features = 1


# In[31]:


model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(length, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# In[32]:


model.summary()


# In[48]:


from tensorflow.keras.callbacks import EarlyStopping


# In[49]:


early_stop = EarlyStopping(monitor='val_loss',patience=2)


# In[50]:


validation_generator = TimeseriesGenerator(scaled_test,scaled_test, length=length, batch_size=1)


# In[53]:


model.fit_generator(generator,epochs=20,
                    validation_data=validation_generator,
                   callbacks=[early_stop])


# In[54]:


losses = pd.DataFrame(model.history.history)


# In[55]:


losses.plot()


# In[56]:


test_predictions = []

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(len(test)):
    
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]
    
    # store prediction
    test_predictions.append(current_pred) 
    
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


# In[57]:


true_predictions=scalar.inverse_transform(test_predictions)


# In[58]:


test['Predictions'] =true_predictions


# In[59]:


test


# In[60]:


test.plot()


# In[61]:


full_scaler = MinMaxScaler()
scaled_full_data = full_scaler.fit_transform(df)


# In[62]:


length = 12
generator = TimeseriesGenerator(scaled_full_data,scaled_full_data,length=length,batch_size=1)


# In[63]:


model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(length, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# fit model
model.fit_generator(generator,epochs=8)


# In[64]:


forecast = []
# Replace periods with whatever forecast length you want
periods = 12

first_eval_batch = scaled_full_data[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(periods):
    
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]
    
    # store prediction
    forecast.append(current_pred) 
    
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


# In[66]:


forecast = scalar.inverse_transform(forecast)


# In[67]:


df


# In[68]:


forecast


# In[69]:


forecast_index = pd.date_range(start='2019-11-01',periods=periods,freq='MS')


# In[70]:


forecast_df = pd.DataFrame(data=forecast,index=forecast_index,
                           columns=['Forecast'])


# In[71]:


forecast_df


# In[72]:


df.plot()
forecast_df.plot()


# In[73]:


ax = df.plot()
forecast_df.plot(ax=ax)


# In[74]:


ax = df.plot()
forecast_df.plot(ax=ax)
plt.xlim('2018-01-01','2020-12-01')


# In[ ]:




