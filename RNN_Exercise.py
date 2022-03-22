#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv('../Data/Frozen_Dessert_Production.csv',index_col='DATE',parse_dates=True)


# In[4]:


df.head()


# **Task: Change the column name to Production**

# In[6]:


df.columns = ['Production']


# In[7]:


df.head()


# In[9]:


df.plot(figsize=(12,8))


# ## Train Test Split

# **TASK: Figure out the length of the data set**

# In[10]:


#CODE HERE


# In[11]:


len(df)


# **TASK: Split the data into a train/test split where the test set is the last 24 months of data.**

# In[12]:


#CODE HERE


# In[13]:


test_size = 24
test_ind = len(df)- test_size


# In[14]:


train = df.iloc[:test_ind]
test = df.iloc[test_ind:]


# In[15]:


len(test)


# ## Scale Data

# **TASK: Use a MinMaxScaler to scale the train and test sets into scaled versions.**

# In[16]:


# CODE HERE


# In[17]:


from sklearn.preprocessing import MinMaxScaler


# In[18]:


scaler = MinMaxScaler()


# In[19]:


# IGNORE WARNING ITS JUST CONVERTING TO FLOATS
# WE ONLY FIT TO TRAININ DATA, OTHERWISE WE ARE CHEATING ASSUMING INFO ABOUT TEST SET
scaler.fit(train)


# In[20]:


scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


# # Time Series Generator
# 
# **TASK: Create a TimeSeriesGenerator object based off the scaled_train data. The batch length is up to you, but at a minimum it should be at least 18 to capture a full year seasonality.**

# In[21]:


#CODE HERE


# In[22]:


from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


# In[23]:


length = 18
n_features=1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=1)


# ### Create the Model
# 
# **TASK: Create a Keras Sequential Model with as many LSTM units you want and a final Dense Layer.**

# In[24]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM


# In[25]:


# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(length, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# In[26]:


model.summary()


# **TASK: Create a generator for the scaled test/validation set. NOTE: Double check that your batch length makes sense for the size of the test set as mentioned in the RNN Time Series video.**

# In[27]:


# CODE HERE


# In[28]:


validation_generator = TimeseriesGenerator(scaled_test,scaled_test, length=length, batch_size=1)


# **TASK: Create an EarlyStopping callback based on val_loss.**

# In[29]:


#CODE HERE


# In[30]:


from tensorflow.keras.callbacks import EarlyStopping


# In[31]:


early_stop = EarlyStopping(monitor='val_loss',patience=2)


# **TASK: Fit the model to the generator, let the EarlyStopping dictate the amount of epochs, so feel free to set the parameter high.**

# In[32]:


# CODE HERE


# In[33]:


# fit model
model.fit_generator(generator,epochs=20,
                    validation_data=validation_generator,
                   callbacks=[early_stop])


# **TASK: Plot the history of the loss that occured during training.**

# In[34]:


# CODE HERE


# In[35]:


loss = pd.DataFrame(model.history.history)
loss.plot()


# ## Evaluate on Test Data
# 
# **TASK: Forecast predictions for your test data range (the last 12 months of the entire dataset). Remember to inverse your scaling transformations. Your final result should be a DataFrame with two columns, the true test values and the predictions.**

# In[36]:


# CODE HERE


# In[37]:


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


# In[38]:


true_predictions = scaler.inverse_transform(test_predictions)


# In[39]:


test['Predictions'] = true_predictions


# In[40]:


test


# **TASK: Plot your predictions versus the True test values. (Your plot may look different than ours).**

# In[41]:


# CODE HERE


# In[42]:


test.plot()


# **TASK: Calculate your RMSE.**

# In[1]:


from sklearn.metrics import mean_squared_error


# In[ ]:


np.sqrt(mean_squared_error(test['Production'],test['Predictions']))


# In[ ]:




