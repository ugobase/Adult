#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Read Csv file
Chris = pd.read_csv('adult.csv')


# In[3]:


#Checking for null values
Chris.isnull().sum()


# In[4]:


#First five rows
Chris.head()


# In[5]:


#Last five rows
Chris.tail()


# In[6]:


#All unique values of workclass column
Chris['workclass'].unique()


# In[7]:


#Data cleaning of workclass column
Chris['workclass'] = Chris['workclass'].replace(['?'], 'Local-gov')
Chris


# In[8]:


#All unique values of occupation column
Chris.occupation.unique()


# In[9]:


#Data cleaning of occupation column
Chris['occupation'] = Chris['occupation'].replace(['?'], 'Sales')


# In[10]:


#All unique values of native-country column
Chris['native-country'].unique()


# In[11]:


#Data cleaning of native-country column
Chris['native-country'] = Chris['native-country'].replace(['?'], 'Peru')


# In[12]:


Chris


# In[13]:


#All information from the file eg columns,index,dtypes
Chris.info()


# In[14]:


#Statistical description of Csv file
Chris.describe()


# In[15]:


#Number of rows
Chris.shape[0]


# In[16]:


#Number of columns
Chris.shape[1]


# In[17]:


#Top 10 largest values of hours-per-week
Chris.nlargest(10,['hours-per-week'])


# In[18]:


#Alternative to previous
Chris.sort_values('hours-per-week', ascending = False).head(10)


# In[19]:


#Top 5 smallest values of age
Chris.nsmallest(5,['age'])


# In[20]:


#Alternative to previous
Chris.sort_values('age').head(5)


# In[21]:


Chris.loc[(Chris['marital-status'] == 'Never-married') & (Chris['gender'] == 'Female') & (Chris['relationship'] == 'Unmarried')].shape[0]


# In[22]:


Chris[(Chris['marital-status'] == 'Never-married') & (Chris['gender'] == 'Female') & (Chris['relationship'] == 'Unmarried')].shape[0]


# In[23]:


#Number of values with race = white and gender = male
Chris.loc[(Chris['race'] == 'White') & (Chris['gender'] == 'Male')].shape[0]


# In[24]:


#Alternative to previous
Chris[(Chris.race == 'White') & (Chris.gender == 'Male')].shape[0]


# In[25]:


Chris.loc[(Chris['race'] == 'Black') & (Chris['gender'] == 'Male') & (Chris['occupation'] == 'Tech-support')].shape[0]


# In[26]:


Chris[(Chris['race'] == 'Black') & (Chris['gender'] == 'Male') & (Chris['occupation'] == 'Tech-support')].shape[0]


# In[27]:


Chris.loc[(Chris['race'] == 'White') & (Chris['gender'] == 'Male') & (Chris['native-country'] == 'Peru')].shape[0]


# In[28]:


Chris[(Chris['race'] == 'White') & (Chris['gender'] == 'Male') & (Chris['native-country'] == 'Peru')].shape[0]


# In[29]:


Chris.occupation.unique()


# In[30]:


Chris.workclass.unique()


# In[31]:


Chris[Chris.workclass.isin(['Without-pay', 'Never-worked'])]


# In[32]:


Chris.loc[(Chris['age'] > 35) & (Chris['workclass'] == 'Federal-gov') & (Chris['hours-per-week'] == 40)].shape[0]


# In[33]:


Chris[Chris.age > 35]['income']


# In[34]:


Chris[Chris.age > 35].income


# In[35]:


Chris.loc[(Chris['age'] == 17) | (Chris['workclass'] == 'Without-pay') & (Chris['hours-per-week'] == 20)].shape[0]


# In[36]:


Chris.loc[(Chris['age'] == 17)].groupby(['workclass']).count().shape[0]


# In[37]:


Chris[Chris.age == 17].workclass.value_counts().shape[0]


# In[38]:


Chris[Chris.age == 17].workclass.value_counts()


# In[39]:


Chris['educational-num'] = Chris['educational-num'].apply(lambda x: x * 1.5)
Chris


# In[40]:


Chris.groupby('age').occupation.describe()


# In[41]:


Chris.groupby('age').income.describe()


# In[42]:


ax = Chris['educational-num'].plot(kind = 'hist', figsize = (8,6))
ax.set_ylabel ('Text')
ax.set_xlabel ('Band')
plt.title('Don', loc = 'right')


# In[43]:


plt.figure(figsize=(12,6))
sns.histplot(Chris['educational-num'])


# In[44]:


ax = Chris['educational-num'].plot(kind = 'box', figsize = (8,6))
ax.set_ylabel ('Next')
ax.set_xlabel ('Band')
plt.title('Don', loc = 'right')


# In[45]:


plt.figure(figsize=(12,6))
sns.boxplot(Chris['educational-num'])


# In[46]:


plt.figure(figsize=(12,6))
sns.boxplot(data = Chris, x = 'race', y = 'educational-num')


# In[47]:


ax = Chris['hours-per-week'].plot(kind = 'density', figsize = (8,6))
ax.set_ylabel ('Next')
ax.set_xlabel ('Band')
plt.title('Don', loc = 'right')


# In[48]:


plt.figure(figsize=(12,8))
sns.displot(Chris['hours-per-week'], kind = 'kde')


# In[49]:


Chris['educational-num'].value_counts().plot(kind = 'bar', figsize = (6,6))


# In[50]:


Chris['race'].value_counts().plot(kind = 'pie', figsize = (6,6))


# In[51]:


plt.figure(figsize = (8,6))
sns.heatmap(Chris.corr(), annot = True, fmt = '0.1f')


# In[52]:


sns.countplot(Chris['race'])


# In[53]:


sns.countplot(Chris['gender'])


# In[ ]:




