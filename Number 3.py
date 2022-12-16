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

#Number of values of never married, female and unmarried
Chris.loc[(Chris['marital-status'] == 'Never-married') & (Chris['gender'] == 'Female') & (Chris['relationship'] == 'Unmarried')].shape[0]


# In[22]:

#Alternative to previous
Chris[(Chris['marital-status'] == 'Never-married') & (Chris['gender'] == 'Female') & (Chris['relationship'] == 'Unmarried')].shape[0]


# In[23]:


#Number of values with race = white and gender = male
Chris.loc[(Chris['race'] == 'White') & (Chris['gender'] == 'Male')].shape[0]


# In[24]:


#Alternative to previous
Chris[(Chris.race == 'White') & (Chris.gender == 'Male')].shape[0]


# In[25]:

#Number of values of black, male and occupation as tech-support
Chris.loc[(Chris['race'] == 'Black') & (Chris['gender'] == 'Male') & (Chris['occupation'] == 'Tech-support')].shape[0]


# In[26]:

#Alternative to previous
Chris[(Chris['race'] == 'Black') & (Chris['gender'] == 'Male') & (Chris['occupation'] == 'Tech-support')].shape[0]


# In[27]:

Number of values of white, male and native country as Peru
Chris.loc[(Chris['race'] == 'White') & (Chris['gender'] == 'Male') & (Chris['native-country'] == 'Peru')].shape[0]


# In[28]:

#Alternative to previous
Chris[(Chris['race'] == 'White') & (Chris['gender'] == 'Male') & (Chris['native-country'] == 'Peru')].shape[0]


# In[29]:

#All the occupations
Chris.occupation.unique()


# In[30]:

#All workclasses
Chris.workclass.unique()


# In[31]:

#Workclass that is without pay or never worked
Chris[Chris.workclass.isin(['Without-pay', 'Never-worked'])]


# In[32]:

#Number of values of age > 35, federal government workclass and 40 hours per work week
Chris.loc[(Chris['age'] > 35) & (Chris['workclass'] == 'Federal-gov') & (Chris['hours-per-week'] == 40)].shape[0]


# In[33]:

#Values of the income of age > 35
Chris[Chris.age > 35]['income']


# In[34]:

#Alternative to previous
Chris[Chris.age > 35].income


# In[35]:

#Number of values of age = 17 or workclass without pay and 20 hours per work week
Chris.loc[(Chris['age'] == 17) | (Chris['workclass'] == 'Without-pay') & (Chris['hours-per-week'] == 20)].shape[0]


# In[36]:

#Number of values of age = 17 grouped by the workclass
Chris.loc[(Chris['age'] == 17)].groupby(['workclass']).count().shape[0]


# In[37]:

#Alternative to previous
Chris[Chris.age == 17].workclass.value_counts().shape[0]


# In[38]:

#Counting and grouping the workclasses of all 17 year olds
Chris[Chris.age == 17].workclass.value_counts()


# In[39]:

#Using lambda function and apply to multiply all the values of educational number by 1.5
Chris['educational-num'] = Chris['educational-num'].apply(lambda x: x * 1.5)
Chris


# In[40]:

#Describing grouping age by occupation
Chris.groupby('age').occupation.describe()


# In[41]:

#Describing grouping age by income
Chris.groupby('age').income.describe()


# In[42]:

#Histogram plot of educational number
ax = Chris['educational-num'].plot(kind = 'hist', figsize = (8,6))
ax.set_ylabel ('Text')
ax.set_xlabel ('Band')
plt.title('Don', loc = 'right')


# In[43]:

#Alternative to previous using seaborn
plt.figure(figsize=(12,6))
sns.histplot(Chris['educational-num'])


# In[44]:

#Box plot of educational number
ax = Chris['educational-num'].plot(kind = 'box', figsize = (8,6))
ax.set_ylabel ('Next')
ax.set_xlabel ('Band')
plt.title('Don', loc = 'right')


# In[45]:

#Alternative to previous using seaborn
plt.figure(figsize=(12,6))
sns.boxplot(Chris['educational-num'])


# In[46]:

#Box plot of race against educational number
plt.figure(figsize=(12,6))
sns.boxplot(data = Chris, x = 'race', y = 'educational-num')


# In[47]:

#Density plot of hours per week
ax = Chris['hours-per-week'].plot(kind = 'density', figsize = (8,6))
ax.set_ylabel ('Next')
ax.set_xlabel ('Band')
plt.title('Don', loc = 'right')


# In[48]:

#Alternative to previous using seaborn
plt.figure(figsize=(12,8))
sns.displot(Chris['hours-per-week'], kind = 'kde')


# In[49]:

#Bar chart of educational number
Chris['educational-num'].value_counts().plot(kind = 'bar', figsize = (6,6))


# In[50]:

#Pie chart of race
Chris['race'].value_counts().plot(kind = 'pie', figsize = (6,6))


# In[51]:

#Correlation of the data
plt.figure(figsize = (8,6))
sns.heatmap(Chris.corr(), annot = True, fmt = '0.1f')


# In[52]:

#Count plot of race
sns.countplot(Chris['race'])


# In[53]:

#Count plot of gender
sns.countplot(Chris['gender'])


# In[ ]:




