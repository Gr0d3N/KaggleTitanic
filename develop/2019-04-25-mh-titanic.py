
# coding: utf-8

# # Intorduction

# ## What is in this notebook?

# ## Inputs

# The following are the inputs which the model needs to run, please select one of the below for each input:

# In[61]:

# inputs go here


# ## Magics & Versions

# The below table shows the version of libraries and packages used for running the model.

# In[62]:

# Inline matplotlib
get_ipython().magic('matplotlib inline')

# Interactive matplotlib plot()
#%matplotlib notebook

# Autoreload packages before runs
# https://ipython.org/ipython-doc/dev/config/extensions/autoreload.html
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# %install_ext http://raw.github.com/jrjohansson/version_information/master/version_information.py
# ~/anaconda/bin/pip install version_information
get_ipython().magic('load_ext version_information')
get_ipython().magic('version_information numpy, scipy, matplotlib, pandas')


# ## Standard imports

# In[63]:

# Standard library
import os
import sys
sys.path.append("../src/")

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Date and time
import datetime
import time

# Ipython imports
from IPython.display import FileLink


# ## Other imports

# In[64]:

# Other imports

# Stats models
import statsmodels.api as sm
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric import smoothers_lowess

from patsy import dmatrices

from sklearn import datasets, svm


# ## Customization

# In[65]:

# Customizations
sns.set() # matplotlib defaults

# Any tweaks that normally go in .matplotlibrc, etc., should explicitly go here
plt.rcParams['figure.figsize'] = (12, 12)

# Silent mode
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)


# In[66]:

# Find the notebook the saved figures came from
fig_prefix = "../figures/2019-04-25-mh-titanic"


# # Data preprocessing

# ## Reading the data

# In[67]:

data = pd.read_csv('../data/training/train.csv')


# In[68]:

data.shape


# We have:
# * 891 rows
# * 12 columns

# ## Exploring the data

# In[69]:

data.head()


# In[70]:

data.dtypes


# The Survived column is the target variable. If Suvival = 1 the passenger survived, otherwise he's dead. The is the variable we're going to predict.
# 
# The other variables describe the passengers. They are the features.
# * PassengerId: and id given to each traveler on the boat
# * Pclass: the passenger class. It has three possible values: 1,2,3 (first, second and third class)
# * The Name of the passeger
# * The Sex
# * The Age
# * SibSp: number of siblings and spouses traveling with the passenger
# * Parch: number of parents and children traveling with the passenger
# * The ticket number
# * The ticket Fare
# * The cabin number
# * The embarkation. This describe three possible areas of the Titanic from which the people embark. Three possible values S,C,Q

# ### Features unique values

# In[71]:

for col in data.columns.values:
    print(col, ' :', data[col].nunique())


# Pclass, Sex, Embarked are categorical features.

# In[72]:

data.describe()


# In[73]:

data.info()


# Age seems to have 177 missing values. let's impute this using the median age.

# ## Missing values

# In[74]:

data['Age'] = data['Age'].fillna(data['Age'].median())


# In[75]:

data.describe()


# # Visualization

# In[76]:

data['Died'] = 1 - data['Survived']


# ## Sex

# In[96]:

# Survival count based on gender
data.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind='Bar', figsize=(12, 7),
                                                          stacked=True, colors=['g', 'r'])
plt.savefig(fig_prefix + '-Sex', dpi=300)


# In[97]:

# Survival ratio based on the gender
data.groupby('Sex').agg('mean')[['Survived', 'Died']].plot(kind='Bar', figsize=(12, 7),
                                                           stacked=True, colors=['g', 'r'])
plt.savefig(fig_prefix + '-Sex-ratio', dpi=300)


# ## Age

# In[98]:

# Violin plots for correlating the survival with sex and age
fig = plt.figure(figsize=(12, 7))
sns.violinplot(x='Sex', y='Age', 
               hue='Survived', data=data,
               split=True,
               palette={0:'r', 1:'g'}
              )
plt.savefig(fig_prefix + '-age', dpi=300)


# As we saw in the chart above and validate by the following:
# * Women survive more than men, as depicted by the larger female green histogram
# 
# Now, we see that:
# * The age conditions the survival for male passengers:
#     * Younger male tend to survive
#     * A large number of passengers between 20 and 40 succumb
# * The age doesn't seem to have a direct impact on the female survival

# ## Ticket fare

# In[104]:

figure = plt.figure(figsize=(25, 12))
plt.hist([data[data['Survived'] == 1]['Fare'], data[data['Survived'] == 0]['Fare']],
         stacked=True, color=['g', 'r'],
         bins=50, label = ['Survived', 'Dead'])
plt.xlabel('Fare')
plt.ylabel('Nubmer of passerngers')
plt.legend()
plt.savefig(fig_prefix + '-fare-dist', dpi=300)


# In[105]:

plt.figure(figsize=(25, 7))
ax = plt.subplot()

ax.scatter(data[data['Survived'] == 1]['Age'], data[data['Survived'] == 1]['Fare'],
           c='g', s=data[data['Survived'] == 1]['Fare'])

ax.scatter(data[data['Survived'] == 0]['Age'], data[data['Survived'] == 0]['Fare'],
           c='r', s=data[data['Survived'] == 0]['Fare'])
plt.savefig(fig_prefix + '-fare-scatter', dpi=300)


# The size of the circles is proportional to the ticket fare.
# 
# On the x-axis, we have the ages and the y-axis, we consider the ticket fare.
# 
# We can observe different clusters:
# 1. Large green dots between x=20 and x=45: adults with the largest ticket fares
# 2. Small red dots between x=10 and x=45, adults from lower classes on the boat
# 3. Small greed dots between x=0 and x=7: these are the children that were saved

# In[107]:

ax = plt.subplot()
ax.set_ylabel('Average fare')
(data.groupby('Pclass')['Fare'].mean()).plot(kind='bar', figsize=(25, 7), ax=ax)
plt.savefig(fig_prefix + '-fare-class', dpi=300)


# ## Embarked

# In[108]:

fig = plt.figure(figsize=(25, 7))
sns.violinplot(x='Embarked', y='Fare', hue='Survived', data=data,
               split=True, palette={0: 'r', 1: 'g'})
plt.savefig(fig_prefix + '-embared-fare', dpi=300)


# It seems that the embarkation C have a wider range of fare tickets and therefore the passengers who pay the highest prices are those who survive.
# 
# We also see this happening in embarkation S and less in embarkation Q.
