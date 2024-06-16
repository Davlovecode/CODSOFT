#!/usr/bin/env python
# coding: utf-8

# # TASK-1: titanic survival predection

# In[1]:


#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#import data set
df = pd.read_csv("F:/codsoft/archive (3)/Titanic-Dataset.csv")


# In[3]:


df.head(10)


# In[4]:


df.shape


# In[5]:


df.describe()


# In[7]:


df['Survived'].value_counts()


# In[11]:


#visualization

sns.countplot(x=df['Survived'],hue=df['Pclass'])


# In[12]:


df["Sex"]


# In[13]:


#visualize wrt gender
sns.countplot(x=df['Sex'],hue=df['Survived'])


# In[14]:


df.groupby('Sex')[['Survived']].mean()


# In[15]:


df['Sex'].unique()


# In[16]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

df['Sex']= labelencoder.fit_transform(df['Sex'])

df.head()


# In[17]:


df['Sex'], df['Survived']


# In[19]:


sns.countplot(x=df['Sex'], hue=df['Survived'])


# In[20]:


df.isna().sum()


# In[21]:


df=df.drop(['Age'],axis=1)


# In[22]:


df_final = df
df_final.head(5)


# In[23]:


x = df[['Pclass','Sex']]
y = df['Survived']


# In[27]:


#model training

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[31]:


from sklearn.linear_model import LogisticRegression

log = LogisticRegression(random_state = 0)
log.fit(x_train, y_train)


# In[32]:


#model prediction

pred = print(log.predict(x_test))


# In[33]:


print(y_test)


# In[34]:


import warnings
warnings.filterwarnings("ignore")

res = log.predict([[2,0]])

if(res==0):
    print("so sorry! not survived")
    
else:
    print("survived")


# In[ ]:




