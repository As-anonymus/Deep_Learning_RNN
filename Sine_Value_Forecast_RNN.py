#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


x = np.linspace(0,50,501)


# In[3]:


x


# In[4]:


y =np.sin(x)


# In[5]:


y


# In[6]:


plt.plot(x,y)


# In[7]:


df = pd.DataFrame(data=y,index=x,columns=['Sine'])


# In[8]:


df


# In[9]:


len(df)


# In[10]:


test_percent = 0.1


# In[11]:


len(df)*test_percent


# In[12]:


test_point = np.round(len(df)*test_percent)


# In[13]:


test_point


# In[14]:


test_ind = int(len(df)-test_point)


# In[15]:


test_ind


# In[16]:


train = df.iloc[:test_ind]


# In[17]:


test = df.iloc[test_ind:]


# In[18]:


train


# In[19]:


from sklearn.preprocessing import MinMaxScaler


# In[20]:


scaler = MinMaxScaler()


# In[21]:


scaler.fit(train)


# In[22]:


scaled_train = scaler.transform(train)


# In[23]:


scaled_test = scaler.transform(test)


# In[24]:


from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


# In[25]:


help(TimeseriesGenerator)


# In[42]:


length = 25
batch_size = 1

generator = TimeseriesGenerator(scaled_train,scaled_train,length=length,batch_size=batch_size)


# In[43]:


len(scaled_train)


# In[44]:


len(generator)


# In[45]:


X,y = generator[0]


# In[46]:


X


# In[47]:


y


# In[48]:


scaled_train


# In[49]:


df.plot()


# In[50]:


length = 50
batch_size = 1

generator = TimeseriesGenerator(scaled_train,scaled_train,length=length,batch_size=batch_size)


# In[51]:


from tensorflow.keras.models import Sequential


# In[52]:


from tensorflow.keras.layers import Dense,SimpleRNN,LSTM


# In[53]:


n_features = 1


# In[55]:


model = Sequential()

model.add(SimpleRNN(50,input_shape=(length,n_features)))

model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')


# In[56]:


model.summary()


# In[57]:


model.fit_generator(generator,epochs=5)


# In[58]:


losses = pd.DataFrame(model.history.history)


# In[60]:


losses.plot()


# In[61]:


first_eval_batch = scaled_train[-length:]


# In[64]:


first_eval_batch = first_eval_batch.reshape((1,length,n_features))


# In[65]:


model.predict(first_eval_batch)


# In[66]:


scaled_test[0]


# In[67]:


test_predictions = []

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1,length,n_features))


# In[68]:


#current_batch


# In[69]:


#predicted_value = [[[99]]]

#np.append(current_batch[:,1:,:],[[[99]]],axis=1)


# In[70]:


test_predictions = []

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1,length,n_features))

for i in range(len(test)):
    current_pred = model.predict(current_batch)[0]
    
    test_predictions.append(current_pred)
    
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


# In[71]:


test_predictions


# In[72]:


true_predictions = scaler.inverse_transform(test_predictions)


# In[74]:


test['Predictions'] = true_predictions


# In[75]:


test


# In[76]:


test.plot()


# In[77]:


from tensorflow.keras.callbacks import EarlyStopping


# In[78]:


early_stop = EarlyStopping(monitor='val_loss',patience=2)


# In[80]:


length = 49

generator = TimeseriesGenerator(scaled_train,scaled_train,length=length,batch_size=1)


validation_genrator = TimeseriesGenerator(scaled_test,scaled_test,length=length,batch_size=1)


# In[81]:


model = Sequential()

model.add(SimpleRNN(50,input_shape=(length,n_features)))

model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')


# In[82]:


model.fit_generator(generator,epochs=20,validation_data=validation_genrator,callbacks=[early_stop])


# In[83]:


test_predictions = []

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1,length,n_features))

for i in range(len(test)):
    current_pred = model.predict(current_batch)[0]
    
    test_predictions.append(current_pred)
    
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


# In[87]:


true_predictions = scaler.inverse_transform(test_predictions)


# In[88]:


test['LSTM Predictions'] = true_predictions
test.plot()


# In[89]:


df.plot()


# In[90]:


full_scaler = MinMaxScaler()
scaled_full_data = full_scaler.fit_transform(df)


# In[91]:


generator = TimeseriesGenerator(scaled_full_data,scaled_full_data,length=length,batch_size=1)


# In[92]:


model = Sequential()

model.add(SimpleRNN(50,input_shape=(length,n_features)))

model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')


# In[93]:


model.fit_generator(generator,epochs=6)


# In[94]:


forecast = []

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1,length,n_features))

for i in range(25):
    current_pred = model.predict(current_batch)[0]
    
    forecast.append(current_pred)
    
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


# In[96]:


forecast = scaler.inverse_transform(forecast)


# In[97]:


forecast


# In[98]:


forecast_index = np.arange(50.1,52.6,step=0.1)


# In[99]:


len(forecast_index)


# In[101]:


plt.plot(df.index,df['Sine'])
plt.plot(forecast_index,forecast)


# In[ ]:




