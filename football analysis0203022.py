#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from scipy import stats
import seaborn as sn
import statsmodels.api as sm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
pd.set_option('display.max_columns',None)
import warnings
warnings.filterwarnings('ignore')
from matplotlib import style
from sklearn.ensemble import RandomForestClassifier


# In[2]:


df=pd.read_csv(r"C:\Users\hisham\Downloads\dataset_players.csv")
df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# Exploratory data analysis

# In[5]:


#age of players
x=df['Age']
plt.figure(figsize=(10,8))
ax=sns.countplot(x,color='#00ffff')
ax.set_xlabel(xlabel='Age of Players',fontsize=16)
ax.set_title(label='Distribution of Age of Players',fontsize=20)
plt.show()


# In[6]:


df['Age'].describe()


# Around 50% of the players are below 23 years old.

# In[7]:


# To show that there are people having same age
# Histogram: number of players's age

sns.set(style = "dark", palette = "colorblind", color_codes = True)
x = df.Age
plt.figure(figsize = (15,8))
plt.style.use('ggplot')
ax = sns.distplot(x, bins = 58, kde = False, color = 'g')
ax.set_xlabel(xlabel = "Player\'s age", fontsize = 16)
ax.set_ylabel(ylabel = 'Number of players', fontsize = 16)
ax.set_title(label = 'Histogram of players age', fontsize = 20)
plt.show()


# In[8]:


#Height of players
plt.figure(figsize = (16, 8))
ax = sns.countplot(x = 'Height', data = df, palette = 'dark')
ax.set_title(label = 'Count of players on Basis of Height', fontsize = 20)
ax.set_xlabel(xlabel = 'Height in Foot per inch', fontsize = 16)
ax.set_ylabel(ylabel = 'Count', fontsize = 16)
plt.show()


# In[9]:


sns.kdeplot(df['Height'])


# Most of the players have height between 175-185

# In[10]:


#weight of players
plt.figure(figsize = (20, 5))
plt.style.available
sns.distplot(df['Weight'], color = 'blue')
plt.title('Different Weights of the Players ', fontsize = 18)
plt.xlabel('Weights associated with the players', fontsize = 16)
plt.ylabel('count of Players', fontsize = 16)
plt.show()


# Almost players have weights between 60-90 and some players with zero weights which is to be replaced

# In[11]:


#Consistency of players
plt.figure(figsize=(6,4),dpi=100)
ax=sns.countplot(x="Consistency",data=df)
ax.set_title("Consistency of players")


# Most of the players have consistency between 7 to 9

# In[12]:


df[df['Consistency'] == 20]['Name']


# There are only 19 players having more Consistency

# In[13]:


#Finding 10 eldest players from the dataset
df.sort_values('Age',ascending=False)[['Name', 'Age', 'NationID', 'PositionsDesc']].head(10).style.background_gradient('inferno')


# In[14]:


#Pair Scatterplots for IntCaps and Professional-Sportsmanship-Temperament
g = sns.pairplot(data= df.query('IntCaps > 0'),y_vars=['IntCaps'],x_vars=['Professional','Sportsmanship','Temperament'],kind="reg")


# We got the plots to inspect the behaviour of the data points between Professional-Sportsmanship-Temperament with International appearances.

# In[15]:


#Correlation between different features


# In[16]:


f,ax = plt.subplots(figsize=(30, 20))
sns.heatmap(df.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.show()


# In[ ]:




