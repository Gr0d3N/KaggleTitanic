
# coding: utf-8

# # Intorduction

# ## What is in this notebook?

# ## Inputs

# The following are the inputs which the model needs to run, please select one of the below for each input:

# In[1]:

# inputs go here


# ## Magics & Versions

# The below table shows the version of libraries and packages used for running the model.

# In[2]:

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

# In[3]:

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

# In[68]:

# Other imports

# Stats models
import statsmodels.api as sm
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric import smoothers_lowess

from patsy import dmatrices

from sklearn import datasets, svm

# Sk-learn
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


# ## Customization

# In[5]:

# Customizations
sns.set() # matplotlib defaults

# Any tweaks that normally go in .matplotlibrc, etc., should explicitly go here
plt.rcParams['figure.figsize'] = (12, 12)

# Silent mode
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)


# In[6]:

# Find the notebook the saved figures came from
fig_prefix = "../figures/2019-04-25-mh-titanic"


# # Data preprocessing

# ## Reading the data

# In[7]:

data = pd.read_csv('../data/training/train.csv')


# In[8]:

data.shape


# We have:
# * 891 rows
# * 12 columns

# ## Exploring the data

# In[9]:

data.head()


# In[10]:

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

# In[11]:

for col in data.columns.values:
    print(col, ' :', data[col].nunique())


# Pclass, Sex, Embarked are categorical features.

# In[12]:

data.describe()


# In[13]:

data.info()


# Age seems to have 177 missing values. let's impute this using the median age.

# ## Missing values

# In[14]:

data['Age'] = data['Age'].fillna(data['Age'].median())


# In[15]:

data.describe()


# # Visualization

# In[16]:

data['Died'] = 1 - data['Survived']


# ## Sex

# In[17]:

# Survival count based on gender
data.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind='Bar', figsize=(12, 7),
                                                          stacked=True, colors=['g', 'r'])
plt.savefig(fig_prefix + '-Sex', dpi=300)


# In[18]:

# Survival ratio based on the gender
data.groupby('Sex').agg('mean')[['Survived', 'Died']].plot(kind='Bar', figsize=(12, 7),
                                                           stacked=True, colors=['g', 'r'])
plt.savefig(fig_prefix + '-Sex-ratio', dpi=300)


# ## Age

# In[19]:

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

# In[20]:

figure = plt.figure(figsize=(25, 12))
plt.hist([data[data['Survived'] == 1]['Fare'], data[data['Survived'] == 0]['Fare']],
         stacked=True, color=['g', 'r'],
         bins=50, label = ['Survived', 'Dead'])
plt.xlabel('Fare')
plt.ylabel('Nubmer of passerngers')
plt.legend()
plt.savefig(fig_prefix + '-fare-dist', dpi=300)


# In[21]:

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

# In[22]:

ax = plt.subplot()
ax.set_ylabel('Average fare')
(data.groupby('Pclass')['Fare'].mean()).plot(kind='bar', figsize=(25, 7), ax=ax)
plt.savefig(fig_prefix + '-fare-class', dpi=300)


# ## Embarked

# In[23]:

fig = plt.figure(figsize=(25, 7))
sns.violinplot(x='Embarked', y='Fare', hue='Survived', data=data,
               split=True, palette={0: 'r', 1: 'g'})
plt.savefig(fig_prefix + '-embared-fare', dpi=300)


# It seems that the embarkation C have a wider range of fare tickets and therefore the passengers who pay the highest prices are those who survive.
# 
# We also see this happening in embarkation S and less in embarkation Q.

# # Feature engineering

# In[24]:

# Function that asserts whether or not a feature has been processed.
def status(feature):
    print('Processing', feature, ': ok')


# In[25]:

# Function for combining the train and the test data
def get_combined_data():
    # reading the train data
    train = pd.read_csv('../data/training/train.csv')
    
    # reading the test data
    test = pd.read_csv('../data/training/test.csv')
    
    # extracting and removing the target from the training data
    targets = train.Survived
    train.drop(['Survived'], 1, inplace=True)
    
    # merging train data and test data for future feature engineering 
    # we'll also remove the PassengerID since this is not an informative feature
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop(['index', 'PassengerId'], inplace=True, axis=1)
    
    return combined


# In[26]:

combined = get_combined_data()
print(combined.shape)


# ## Passenger titles

# In[27]:

titles = set()
for name in data['Name']:
    titles.add(name.split(',')[1].split('.')[0].strip())


# In[28]:

print(titles)


# In[29]:

title_dic ={
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}


# In[30]:

def get_titles():
    # We extract title from each name
    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
    # a map of more aggregated title
    # we map each title
    combined['Title'] = combined.Title.map(title_dic)
    status('Title')
    
    return combined


# In[31]:

combined = get_titles()


# In[32]:

combined.head()


# In[33]:

# Checking
combined[combined.Title.isnull()]


# There is indeed a NaN value in the line 1305. In fact the corresponding name is Oliva y Ocana, Dona. Fermina.
# 
# This title was not encoutered in the train dataset.

# ## Passenger ages

# In[34]:

# Number of missing ages in train set
print(combined.iloc[:891].Age.isnull().sum())


# In[35]:

# Number of missing ages in test set
print(combined.iloc[891:].Age.isnull().sum())


# In[36]:

# Grouping
grouped_train = combined.iloc[:891].groupby(['Sex', 'Pclass', 'Title'])
grouped_median_train = grouped_train.median()
grouped_median_train = grouped_median_train.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]
grouped_median_train.head()


# In[37]:

def fill_age(row):
    condition = (
        (grouped_median_train['Sex'] == row['Sex']) & 
        (grouped_median_train['Title'] == row['Title']) & 
        (grouped_median_train['Pclass'] == row['Pclass'])
    ) 
    return grouped_median_train[condition]['Age'].values[0]


def process_age():
    global combined
    # a function that fills the missing values of the Age variable
    combined['Age'] = combined.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
    status('age')
    return combined


# In[38]:

combined = process_age()


# In[39]:

combined.head()


# ## Names

# In[40]:

def process_names():
    global combined
    # we clean the Name variable
    combined.drop('Name', axis=1, inplace=True)
    
    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')
    combined = pd.concat([combined, titles_dummies], axis=1)
    
    # removing the title variable
    combined.drop('Title', axis=1, inplace=True)
    
    status('names')
    return combined


# In[41]:

combined = process_names()


# In[42]:

combined.head()


# ## Fare

# In[43]:

def process_fares():
    global combined
    # there's one missing fare value - replacing it with the mean.
    combined.Fare.fillna(combined.iloc[:891].Fare.mean(), inplace=True)
    status('fare')
    return combined


# In[44]:

combined = process_fares()


# ## Embarked

# In[45]:

def process_embarked():
    global combined
    # two missing embarked values - filling them with the most frequent one in the train  set(S)
    combined.Embarked.fillna('S', inplace=True)
    # dummy encoding 
    embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
    combined = pd.concat([combined, embarked_dummies], axis=1)
    combined.drop('Embarked', axis=1, inplace=True)
    status('embarked')
    return combined


# In[46]:

combined = process_embarked()


# In[47]:

combined.head()


# ## Cabin

# In[48]:

train_cabin, test_cabin = set(), set()

for c in combined.iloc[:891]['Cabin']:
    try:
        train_cabin.add(c[0])
    except:
        train_cabin.add('U')
        
for c in combined.iloc[891:]['Cabin']:
    try:
        test_cabin.add(c[0])
    except:
        test_cabin.add('U')


# In[49]:

print(train_cabin)


# In[50]:

print(test_cabin)


# We don't have any cabin letter in the test set that is not present in the train set.

# In[51]:

def process_cabin():
    global combined    
    # replacing missing cabins with U (for Uknown)
    combined.Cabin.fillna('U', inplace=True)
    
    # mapping each Cabin value with the cabin letter
    combined['Cabin'] = combined['Cabin'].map(lambda c: c[0])
    
    # dummy encoding ...
    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')    
    combined = pd.concat([combined, cabin_dummies], axis=1)

    combined.drop('Cabin', axis=1, inplace=True)
    status('cabin')
    return combined


# This function replaces NaN values with U (for Unknow). It then maps each Cabin value to the first letter. Then it encodes the cabin values using dummy encoding again.

# In[52]:

combined = process_cabin()


# In[53]:

combined.head()


# ## Sex

# In[54]:

def process_sex():
    global combined
    # mapping string values to numerical one 
    combined['Sex'] = combined['Sex'].map({'male':1, 'female':0})
    status('Sex')
    return combined


# In[55]:

combined = process_sex()


# ## Pclass

# In[56]:

def process_pclass():
    
    global combined
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")
    
    # adding dummy variable
    combined = pd.concat([combined, pclass_dummies],axis=1)
    
    # removing "Pclass"
    combined.drop('Pclass',axis=1,inplace=True)
    
    status('Pclass')
    return combined


# In[57]:

combined = process_pclass()


# ## Ticket

# In[58]:

def cleanTicket(ticket):
    ticket = ticket.replace('.', '')
    ticket = ticket.replace('/', '')
    ticket = ticket.split()
    ticket = map(lambda t : t.strip(), ticket)
    ticket = list(filter(lambda t : not t.isdigit(), ticket))
    if len(ticket) > 0:
        return ticket[0]
    else: 
        return 'XXX'


# In[59]:

tickets = set()
for t in combined['Ticket']:
    tickets.add(cleanTicket(t))


# In[60]:

print(len(tickets))


# In[61]:

def process_ticket():
    
    global combined
    
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = map(lambda t : t.strip(), ticket)
        ticket = list(filter(lambda t : not t.isdigit(), ticket))
        if len(ticket) > 0:
            return ticket[0]
        else: 
            return 'XXX'
    

    # Extracting dummy variables from tickets:

    combined['Ticket'] = combined['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(combined['Ticket'], prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies], axis=1)
    combined.drop('Ticket', inplace=True, axis=1)

    status('Ticket')
    return combined


# In[62]:

combined = process_ticket()


# ## Family
# This part includes creating new variables based on the size of the family (the size is by the way, another variable we create).
# 
# This creation of new variables is done under a realistic assumption: Large families are grouped together, hence they are more likely to get rescued than people traveling alone.

# In[63]:

def process_family():
    
    global combined
    # introducing a new feature : the size of families (including the passenger)
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    
    # introducing other features based on the family size
    combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
    
    status('family')
    return combined


# In[64]:

combined = process_family()


# In[65]:

print(combined.shape)


# In[67]:

combined.head()

